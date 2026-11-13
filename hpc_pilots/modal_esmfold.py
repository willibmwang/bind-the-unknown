"""ESMFold on a Modal GPU, speaking the same protocol as the public Atlas host.

**Why this exists.** `folded_structure.ENDPOINT` defaults to `api.esmatlas.com`, a keyless
public service whose ceiling is 400 residues -- measured, not documented: 400 returns 200 and
401 returns HTTP 413. That ceiling is the host's and not ESMFold's, and on the held-out
SARS-CoV-2 run it refuses eight of the twelve shortlisted chains whole, including all three
spike chains at 458 to 641 residues, the polymerase at 932 and nsp3 at 1,922. Seven of the
eight are rescued by folding their annotated regions instead, which is a real result and also
a weaker one: a region folded alone has lost whatever its neighbours in the chain contributed
to its shape, and the chain record says so in its own limitations.

This serves the same model with no such ceiling, so those chains can be folded whole.

**It is a drop-in and nothing here knows about it.** `_attempt` POSTs the raw sequence as an
ASCII body with `Content-Type: text/plain` and expects PDB text back; that is exactly what
this answers, so pointing the lane at a GPU is two environment variables and no code change:

    export ESMFOLD_ENDPOINT="https://<workspace>--esmfold-fold.modal.run"
    export ESMFOLD_MAX_RESIDUES=2000

`folded_structure.citation` already branches on the host, so a fold produced here is not
labelled as the Atlas's. That matters more than it looks: the two are the same weights but not
the same artifact, and `runs/folded_structure/mpro_nsp5.pdb` -- which this repository
reproduces byte for byte against the public host -- should not be expected to reproduce
against this one. Different hardware, different kernels, different rounding.

**Cost shape.** The class holds the model in GPU memory between calls and scales to zero after
`SCALEDOWN`, so an idle deployment costs nothing and a run of a dozen chains pays one cold
start plus the folds. Weights live in a Volume rather than the image, so the 11 GB download
happens once per volume rather than once per image build.

Deploy with:

    modal deploy -e <environment> hpc_pilots/modal_esmfold.py
"""

import os

import modal

MODEL = "facebook/esmfold_v1"
GPU = os.environ.get("ESMFOLD_MODAL_GPU", "H100")
"""An H100, for its 80 GB rather than its speed.

Length is what decides this, not throughput: attention over the trunk is quadratic, and the
longest chain here is nsp3 at 1,922 residues against a polymerase at 932 and a spike at 641.
A 40 GB card folds the spike comfortably and is the first thing to exhaust on nsp3, so the
memory is the capability and the speed is a bonus.

**Not a B200, and the reason is the pin.** Blackwell needs CUDA 12.8 and torch 2.7 or newer,
so moving to one means moving off `torch==2.5.1` -- and the pin is not bureaucracy here, it
is what stops the fold output drifting under a stack upgrade nobody asked for. ESMFold at
these lengths does not need the extra silicon, so a B200 would buy a few seconds a fold and
cost a change to what the folds are. If a future target makes it worth it, bump the torch pin
in the same commit and say in the message that the structures moved.
"""

SCALEDOWN = 300
TIMEOUT = 1800

image = (
    modal.Image.debian_slim(python_version="3.11")
    # Pinned, because an unpinned transformers is a silent change to what the folds are.
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "accelerate==1.1.1",
        "einops==0.8.0",
        "scipy==1.14.1",
        "fastapi[standard]==0.115.5",
    )
    .env({"HF_HOME": "/cache"})
)

cache = modal.Volume.from_name("esmfold-weights", create_if_missing=True)
app = modal.App("esmfold")

SECRETS = ([modal.Secret.from_name(os.environ["ESMFOLD_MODAL_SECRET"])]
           if os.environ.get("ESMFOLD_MODAL_SECRET") else [])
"""The shared secret holding `ESMFOLD_TOKEN`, when one has been created.

Optional on purpose. A deployment with no secret serves anyone who has the URL, which is what
a first bring-up produces and is fine while the only caller is a test; a deployment with one
refuses every request that does not carry the token. Making it optional means the lock can be
added later without a code change, and means this file does not fail to deploy for a secret
that does not exist yet. Set `ESMFOLD_MODAL_SECRET=esmfold-auth` at deploy time to switch it
on. The token itself is never in this repository.
"""


@app.cls(
    image=image,
    gpu=GPU,
    volumes={"/cache": cache},
    secrets=SECRETS,
    timeout=TIMEOUT,
    scaledown_window=SCALEDOWN,
)
class Folder:
    """One warm ESMFold per container."""

    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoTokenizer, EsmForProteinFolding

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        model = EsmForProteinFolding.from_pretrained(MODEL, low_cpu_mem_usage=True)
        model = model.cuda().eval()
        # The language-model half in fp16 and the folding trunk left alone: the recipe the
        # model card gives for fitting long sequences without moving the output.
        model.esm = model.esm.half()
        # Chunked attention over the trunk trades speed for memory. 64 was the cautious
        # setting carried over from 40 GB cards and it is too cautious here: on an 80 GB
        # H100 a 1,922-residue chain at chunk 64 ran past 150 seconds, which is where a
        # Modal web endpoint stops holding the connection and answers 303 with a polling
        # URL -- and that URL cannot be followed from a POST, so the fold is lost rather
        # than slow. The limit this has to beat is a wall-clock one, not a memory one, so
        # the chunk goes up to spend the headroom the card actually has.
        chunk = os.environ.get("ESMFOLD_CHUNK", "128")
        model.trunk.set_chunk_size(None if chunk.lower() in ("", "0", "none") else int(chunk))
        self.model = model
        cache.commit()

    def _fold(self, body):
        """The whole contract as a pure function of the request body. Returns (status, text)."""
        sequence = "".join(body.decode("ascii", "replace").split())
        if not sequence:
            return 400, "empty sequence"
        # Mirrors the public host's own refusals rather than inventing new ones, so the
        # lane's existing handling of a 413 and a 422 keeps working against this deployment.
        ceiling = int(os.environ.get("ESMFOLD_MODAL_CEILING") or 2700)
        if len(sequence) > ceiling:
            return 413, f"Sequence is longer than {ceiling}."
        bad = sorted(set(sequence) - set("ACDEFGHIKLMNPQRSTVWY"))
        if bad:
            return 422, f"sequence holds {''.join(bad)!r}, which ESMFold does not accept"
        try:
            with self.torch.no_grad():
                return 200, self.model.infer_pdb(sequence)
        except Exception as error:                            # noqa: BLE001
            # Running out of memory on a long chain is the expected failure and is reported
            # as its own status, not as a generic 500 a caller would retry three times.
            name = type(error).__name__
            oom = "OutOfMemory" in name or "out of memory" in str(error).lower()
            return (507 if oom else 500), f"{name}: {error}"

    @modal.asgi_app()
    def web(self):
        """A bare ASGI callable: POST a raw sequence, get a PDB. The Atlas host's contract.

        **Raw ASGI rather than a typed FastAPI handler, and that is not a style choice.**
        A `fastapi_endpoint` reads its request from a parameter annotated `Request`, and
        fastapi is not importable on the machine this file is deployed from -- so the import
        has to be deferred into the container and the annotation becomes a string that
        FastAPI resolves against module globals at startup. When that resolution does not
        happen, FastAPI does not fail: it decides `request` is a QUERY PARAMETER and answers
        every POST with 422 before the model is asked. That is the quietest possible failure,
        because 422 is a status the fold lane already treats as a real refusal from the host,
        so a pipeline pointed here would have recorded every chain as a capability absence
        rather than as a bug. Three lines of ASGI need no annotations and cannot do that.
        """
        async def app(scope, receive, send):
            if scope["type"] != "http":                       # pragma: no cover
                return
            body, more = b"", True
            while more:
                message = await receive()
                body += message.get("body", b"")
                more = message.get("more_body", False)
            wanted = os.environ.get("ESMFOLD_TOKEN")
            if wanted:
                headers = {k.decode().lower(): v.decode()
                           for k, v in scope.get("headers") or ()}
                if headers.get("authorization") != f"Bearer {wanted}":
                    status, text = 401, "missing or wrong bearer token"
                else:
                    status, text = self._fold(body)
            else:
                status, text = self._fold(body)
            payload = text.encode("utf-8")
            await send({"type": "http.response.start", "status": status,
                        "headers": [(b"content-type", b"text/plain; charset=utf-8"),
                                    (b"content-length", str(len(payload)).encode())]})
            await send({"type": "http.response.body", "body": payload})

        return app
