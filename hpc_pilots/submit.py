"""Submit dependency-gated pilots; write each accepted ID immediately."""
import argparse
import json
import shutil
import subprocess

from .common import ROOT, config, digest, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--partition', default='compute')
    parser.add_argument('--account')
    parser.add_argument('--with-agent', action='store_true')
    parser.add_argument('--add-agent', action='store_true', help='Add the GPU phase to a recorded baseline submission')
    parser.add_argument('--concurrency', type=int, default=4)
    args = parser.parse_args()
    if not shutil.which('sbatch'):
        raise SystemExit('No Slurm submission command on this host')
    status_path = ROOT / 'runs/submissions.json'
    if status_path.exists() and not args.add_agent:
        raise SystemExit('Submission registry already exists; inspect it before launching another sweep')
    if args.with_agent or args.add_agent:
        manifest = ROOT / 'models/manifest.json'
        if not manifest.exists() or json.loads(manifest.read_text()).get('status') != 'staged':
            raise SystemExit('Pinned local LLM weights must be fully staged before GPU submission')
    if args.add_agent:
        if not status_path.exists():
            raise SystemExit('No baseline submission registry')
        status = json.loads(status_path.read_text())
        if 'mats-agent' in status['jobs']:
            raise SystemExit('GPU phase already submitted')
        if 'mats-baseline-summary' not in status['jobs']:
            raise SystemExit('Baseline pipeline submission is incomplete')
    else:
        status = {'config_sha256': digest(ROOT / 'hpc_pilots/configs/today.json'), 'jobs': {},
                  'agent_included': args.with_agent, 'status': 'submitting'}
    (ROOT / 'runs/logs').mkdir(parents=True, exist_ok=True)
    write_json(status_path, status)

    def submit(name, script, cpus, memory, walltime, extra=(), command=(), dependency=None):
        argv = ['sbatch', '--parsable', '--partition', args.partition, '--job-name', name,
                '--chdir', str(ROOT), '--cpus-per-task', str(cpus), '--mem', memory,
                '--time', walltime, '--output', str(ROOT / 'runs/logs' / (name + '_%A_%a.out'))]
        if args.account:
            argv += ['--account', args.account]
        if dependency:
            argv += ['--dependency', 'afterok:' + ':'.join(str(i) for i in dependency)]
        argv += list(extra) + [str(ROOT / 'hpc_pilots/slurm' / script)] + list(command)
        job = subprocess.check_output(argv, text=True).strip().split(';')[0]
        if not job.isdigit():
            raise RuntimeError('Unexpected sbatch job identifier')
        status['jobs'][name] = {'id': job, 'command': argv, 'state': 'submitted'}
        write_json(status_path, status)
        print(name, job, flush=True)
        return job

    if args.add_agent:
        submit('mats-agent', 'agent.sbatch', 8, '24G', '04:00:00', extra=['--gres', 'gpu:l40:1'],
               dependency=[status['jobs']['mats-baseline-summary']['id']])
        status.update(status='submitted', agent_included=True)
        write_json(status_path, status)
        return

    smoke = submit('mats-smoke', 'smoke.sbatch', 2, '4G', '00:30:00')
    bc = config()['binding']
    n = bc['evaluation_pools'] * len(bc['feedback_delays']) * (len(bc['policies']) - 1)
    binding = submit('mats-binding', 'run.sbatch', 1, '2G', '02:00:00',
                     extra=['--array', f'0-{n-1}%{args.concurrency}'],
                     command=['binding-array'], dependency=[smoke])
    atlas = submit('mats-atlas', 'atlas.sbatch', 4, '4G', '04:00:00',
                   extra=['--array', '0-11%2'], dependency=[smoke])
    structure = submit('mats-structure', 'run.sbatch', 1, '2G', '00:30:00',
                       extra=['--array', '0-2%2'], command=['structure-array'], dependency=[atlas])
    baseline_summary = submit('mats-baseline-summary', 'run.sbatch', 1, '2G', '00:10:00',
                              command=['summarize'], dependency=[binding, structure])
    if args.with_agent:
        submit('mats-agent', 'agent.sbatch', 8, '24G', '04:00:00', extra=['--gres', 'gpu:l40:1'],
               dependency=[baseline_summary])
    status['status'] = 'submitted'
    write_json(status_path, status)


if __name__ == '__main__':
    main()
