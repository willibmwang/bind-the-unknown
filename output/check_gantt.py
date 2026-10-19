"""Local layout/interaction check for the dated Gantt fragment."""
from pathlib import Path
import json
from playwright.sync_api import sync_playwright

root=Path(__file__).resolve().parent
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless=True)
    page=browser.new_page(viewport={'width':736,'height':1000},device_scale_factor=1)
    errors=[]
    page.on('pageerror',lambda error:errors.append(str(error)))
    page.goto((root/'emergency-scientist-detailed-gantt-preview.html').as_uri())
    frame=page.frame_locator('iframe')
    frame.locator('#emergency-scientist-gantt').wait_for()
    page.wait_for_timeout(300)
    assert frame.locator('.task-row').count()==82
    assert frame.locator('details[open]').count()==1
    frame.get_by_role('button',name='TRAIN-11 · W8D5: produce development-only high-budget targets',exact=True).click()
    assert frame.locator('[data-detail]').is_visible()
    assert 'eight trajectories' in frame.locator('[data-detail]').inner_text()
    assert 'Nov 20' in frame.locator('[data-detail]').inner_text()
    frame.get_by_role('button',name='TRAIN-03 · W8D2: checkpoint resume, sampling and idempotent jobs',exact=True).click()
    assert 'Nov 17' in frame.locator('[data-detail]').inner_text()
    iframe=page.frames[1]
    checks=[]
    for width in (736,360,320):
        page.set_viewport_size({'width':width,'height':1000})
        page.wait_for_timeout(100)
        geometry=iframe.evaluate('''() => {
          const root=document.getElementById('emergency-scientist-gantt');
          return {width:innerWidth,scroll:document.documentElement.scrollWidth,
            overflow:[...root.querySelectorAll('button,.axis,.lane,dd')].filter(e=>e.getClientRects().length && e.getBoundingClientRect().right>innerWidth+1).map(e=>e.textContent.slice(0,60))};
        }''')
        assert geometry['scroll']<=geometry['width']+1,geometry
        assert not geometry['overflow'],geometry
        checks.append(geometry)
        # Paint the whole iframe for visual QA, rather than only its viewport.
        full_height=iframe.evaluate('document.documentElement.scrollHeight')+32
        page.set_viewport_size({'width':width,'height':full_height})
        iframe.evaluate('window.scrollTo(0,0)')
        page.evaluate('window.scrollTo(0,0)')
        page.wait_for_timeout(100)
        page.screenshot(path=str(root/f'gantt-check-{width}.png'),full_page=True)
    assert not errors,errors
    print(json.dumps({'tasks':82,'interaction':'pass','width_checks':checks,'javascript_errors':errors},indent=2))
    browser.close()
