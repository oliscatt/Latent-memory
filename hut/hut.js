/* 山屋各页之间的总线。
   页面单独打开时什么都不做（按钮照旧）；装进全景（prototype.html）时，返回、回到那一天、这条记错了、去地图看都交给全景去切页。
   hut.go('cassette', {date:'2026-06-18'})  推到磁带那一天
   hut.go('lock', {quote, date, kind:'wrong'|'changed'})  写给 TA，便利贴上先印好是哪一条
   hut.go('map')  去地图
   hut.onShow(data => …)  全景切过来时带的东西 */
(() => {
  /* 只认全景给的那个框（name=hut）：artifact 本身也套在别人的框里，不能只看有没有 parent */
  const shell = window.parent !== window && window.name === 'hut' ? window.parent : null;
  const hooks = [];
  let pending = null;
  window.hut = {
    inShell: !!shell,
    go(page, data = {}){ if (shell) shell.postMessage({hut:'go', page, data}, '*'); return !!shell; },
    back(){ if (shell) shell.postMessage({hut:'back'}, '*'); return !!shell; },
    onShow(f){ hooks.push(f); if (pending) f(pending); },
  };
  addEventListener('message', e => {
    if (!shell || e.source !== shell || !e.data || e.data.hut !== 'show') return;
    pending = e.data.data || {};
    hooks.forEach(f => f(pending));
  });
  /* 返回箭头：在全景里点它就是退回全景 */
  /* 真数据：同一个 server 挂着 /admin/api（mcp_server --admin-token --ui）就用，拿不到回 null，
     页面接着用自己的示例数据——artifact 里那份原型照样能看。
     钥匙只在最外层问一次，只存在这台设备上；框里的页不问。拿到的文字各页一律 esc 了再放。 */
  const KEY = 'hut-admin-key';
  const keyGet = () => { try { return localStorage.getItem(KEY); } catch (e) { return null; } };
  const keySet = v => { try { localStorage.setItem(KEY, v); } catch (e) {} };
  /* 录屏用：data/demo.js 在就先用它（双击打开也能读），不在就照旧 */
  let demo;
  const demoData = () => demo || (demo = new Promise(ok => {
    const s = document.createElement('script');
    s.src = 'data/demo.js';
    s.onload = () => ok(window.HUT_DEMO || null);
    s.onerror = () => ok(null);
    document.head.appendChild(s);
  }));
  /* hut.api(path) 读；hut.api(path, body) 写——页面口子能写的只有贴便条（POST /notes），演示数据下不写 */
  window.hut.api = async (path, body) => {
    const d = body === undefined ? await demoData() : null;
    if (d && d[path] !== undefined){
      document.querySelectorAll('.sample').forEach(e => { e.hidden = true; });
      return d[path];
    }
    if (!/^https?:$/.test(location.protocol)) return null;
    let key = keyGet();
    for (let i = 0; i < 2; i++){
      let r;
      const auth = key ? {Authorization:'Bearer ' + key} : {};
      try { r = await fetch('/admin/api' + path, body === undefined ? {headers:auth, cache:'no-store'}
        : {method:'POST', headers:{...auth, 'Content-Type':'application/json'}, body:JSON.stringify(body), cache:'no-store'}); }
      catch (e) { return null; }
      if (r.status === 401 && i === 0 && !shell){
        key = (window.prompt('山屋的钥匙（起服务时的 --admin-token）') || '').trim();
        if (!key) return null;
        keySet(key);
        continue;
      }
      if (!r.ok) return null;
      document.querySelectorAll('.sample').forEach(e => { e.hidden = true; });   // 用的是真数据，"sample data"那行收起来
      return r.json();
    }
    return null;
  };
  /* 录屏（?rec）：各页的 "sample data" 都不露——小锁信这一期不接数据，那行原本一直在 */
  try {
    if (/[?&]rec(=|&|$)/.test(window.top.location.search)){
      const hide = () => document.querySelectorAll('.sample').forEach(e => { e.hidden = true; });
      if (document.readyState === 'loading') addEventListener('DOMContentLoaded', hide); else hide();
    }
  } catch (e) {}
  addEventListener('click', e => {
    if (!shell) return;
    const b = e.target.closest && e.target.closest('button.back,button.goback'); if (!b) return;
    e.preventDefault(); e.stopPropagation(); hut.back();
  }, true);
})();
