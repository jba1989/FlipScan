// Display-language switch for the GUI and the reader.
//
// UI text is written in Traditional Chinese in the source and doubles as the
// catalog key: t("關閉") for a plain string, T`已匯入 ${n} 頁` for a template
// (key "已匯入 {0} 頁"). English lives in i18n.en.js; a missing entry falls
// back to the Chinese source. An English template may pick a plural form with
// {N?one|many}, chosen by whether value N is 1 — e.g. "{0} page{0?|s}".
//
// The language comes from the viewer's choice (localStorage), else the
// browser: zh* → 繁體中文, anything else → English. Switching reloads the page.
(() => {
  const STORE = "flipscan.lang";
  const LANGS = { "zh-Hant": "繁體中文", en: "English" };

  function detect() {
    try {
      const saved = localStorage.getItem(STORE);
      if (saved in LANGS) return saved;
    } catch {}
    const first = (navigator.languages && navigator.languages[0]) || navigator.language || "";
    return /^zh/i.test(first) ? "zh-Hant" : "en";
  }

  const lang = detect();
  const catalog = lang === "en" ? (window.FLIPSCAN_EN || {}) : {};
  const has = (k) => Object.prototype.hasOwnProperty.call(catalog, k);
  document.documentElement.lang = lang;

  const fill = (tpl, vals) =>
    tpl.replace(/\{(\d+)(?:\?([^|{}]*)\|([^{}]*))?\}/g, (_, i, one, many) =>
      one === undefined ? String(vals[+i]) : (Number(vals[+i]) === 1 ? one : many));

  function t(s) { return has(s) ? catalog[s] : s; }

  function T(strings, ...vals) {
    if (lang !== "en") return String.raw({ raw: strings }, ...vals);
    let key = strings[0];
    for (let i = 0; i < vals.length; i++) key += `{${i}}` + strings[i + 1];
    return has(key) ? fill(catalog[key], vals) : String.raw({ raw: strings }, ...vals);
  }

  // static markup: text nodes and a few attributes, matched on the
  // whitespace-normalized text so source line breaks don't matter
  const ATTRS = ["title", "placeholder", "aria-label", "alt"];
  const norm = (s) => s.replace(/\s+/g, " ").trim();
  function translateStatic(root = document.body) {
    if (lang !== "en" || !root) return;
    document.title = t(document.title);
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    for (let n = walker.nextNode(); n; n = walker.nextNode()) {
      const tag = n.parentNode && n.parentNode.nodeName;
      if (tag === "SCRIPT" || tag === "STYLE") continue;
      const k = norm(n.nodeValue);
      if (k && has(k)) n.nodeValue = n.nodeValue.replace(n.nodeValue.trim(), catalog[k]);
    }
    for (const el of root.querySelectorAll(ATTRS.map((a) => `[${a}]`).join(","))) {
      for (const a of ATTRS) {
        const v = el.getAttribute(a);
        if (v && has(norm(v))) el.setAttribute(a, catalog[norm(v)]);
      }
    }
  }

  function setLang(next) {
    if (!(next in LANGS) || next === lang) return;
    try { localStorage.setItem(STORE, next); } catch {}
    location.reload();
  }

  // the server renders log lines and error details in the viewer's language
  function reportToServer() {
    return fetch("/api/language", {
      method: "PUT", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ lang }),
    }).catch((e) => console.warn("language sync failed:", e));
  }

  // a <select> for the language picker, current language preselected
  function picker() {
    return `<select onchange="I18N.setLang(this.value)" title="${t("語言")}"
      style="width:100%;margin:0;padding:.3rem .4rem;font-size:.8rem">${
      Object.entries(LANGS).map(([k, name]) =>
        `<option value="${k}"${k === lang ? " selected" : ""}>${name}</option>`).join("")
    }</select>`;
  }

  window.t = t;
  window.T = T;
  window.I18N = { lang, LANGS, setLang, translateStatic, reportToServer, picker };
})();
