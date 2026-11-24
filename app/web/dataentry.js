// Admin Studio for the knowledge base — vanilla ES module, no build step.
// Security: user data is only ever rendered through textContent / DOM properties (see h()),
// never innerHTML. innerHTML is used solely for the static SVG icons below.

const API = document.body.dataset.api || "/api/v1";
const ADMIN = `${API}/admin`;
const $ = (selector, root = document) => root.querySelector(selector);

// ---------------------------------------------------------------- DOM helpers

function h(tag, props = {}, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value == null || value === false) continue;
    if (key === "class") el.className = value;
    else if (key.startsWith("on")) el.addEventListener(key.slice(2), value);
    else if (key in el) el[key] = value;
    else el.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child != null && child !== false) el.append(child instanceof Node ? child : String(child));
  }
  return el;
}

const ICONS = {
  database: '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/><path d="M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/>',
  refresh: '<path d="M21 12a9 9 0 0 1-15.5 6.2L3 16"/><path d="M3 21v-5h5"/><path d="M3 12a9 9 0 0 1 15.5-6.2L21 8"/><path d="M21 3v5h-5"/>',
  reload: '<path d="M21 12a9 9 0 1 1-3-6.7L21 8"/><path d="M21 3v5h-5"/>',
  message: '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>',
  theme: '<path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/>',
  logout: '<path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><path d="m16 17 5-5-5-5"/><path d="M21 12H9"/>',
  upload: '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="m17 8-5-5-5 5"/><path d="M12 3v12"/>',
  search: '<circle cx="11" cy="11" r="7"/><path d="m21 21-4.3-4.3"/>',
  file: '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/>',
  close: '<path d="M18 6 6 18"/><path d="m6 6 12 12"/>',
  send: '<path d="m22 2-7 20-4-9-9-4Z"/><path d="M22 2 11 13"/>',
  trash: '<path d="M3 6h18"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/><path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>',
  check: '<path d="M20 6 9 17l-5-5"/>',
  alert: '<circle cx="12" cy="12" r="10"/><path d="M12 8v4"/><path d="M12 16h.01"/>',
  info: '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',
  link: '<path d="M15 3h6v6"/><path d="M10 14 21 3"/><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>',
  plus: '<path d="M12 5v14"/><path d="M5 12h14"/>',
  save: '<path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/><path d="M17 21v-8H7v8"/><path d="M7 3v5h8"/>',
  up: '<path d="m18 15-6-6-6 6"/>',
  down: '<path d="m6 9 6 6 6-6"/>',
};

function icon(name, size = 16) {
  const span = h("span", { class: "inline-flex shrink-0", "aria-hidden": "true" });
  span.innerHTML = `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${ICONS[name]}</svg>`;
  return span;
}

document.querySelectorAll("[data-icon]").forEach((el) => el.prepend(icon(el.dataset.icon, el.classList.contains("h-10") ? 20 : 16)));

const rtf = new Intl.RelativeTimeFormat(undefined, { numeric: "auto" });
function ago(iso) {
  const seconds = (new Date(iso) - Date.now()) / 1000;
  for (const [unit, size] of [["year", 31536000], ["month", 2592000], ["day", 86400], ["hour", 3600], ["minute", 60]]) {
    if (Math.abs(seconds) >= size) return rtf.format(Math.round(seconds / size), unit);
  }
  return "just now";
}

const fmtMs = (ms) => (ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`);
const label = (category) => category.replaceAll("_", " ");

function safeUrl(value) {
  try {
    const url = new URL(value);
    return url.protocol === "http:" || url.protocol === "https:" ? url.href : null;
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------- API

class ApiError extends Error {
  constructor(status, message, details = []) {
    super(message);
    this.status = status;
    this.details = details;
  }
}

async function api(path, { method = "GET", body } = {}) {
  let response;
  try {
    response = await fetch(path, {
      method,
      credentials: "same-origin",
      headers: body === undefined ? {} : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    throw new ApiError(0, "Network error — is the server running?");
  }
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    if (response.status === 401 && path.startsWith(ADMIN) && !path.endsWith("/auth")) {
      showLogin("Your session has ended. Please sign in again.");
    }
    throw new ApiError(response.status, data?.error || `Request failed (HTTP ${response.status}).`, data?.details || []);
  }
  return data;
}

// ---------------------------------------------------------------- toasts

const TOAST_TONES = {
  success: ["check", "text-emerald-600 dark:text-emerald-400"],
  error: ["alert", "text-red-600 dark:text-red-400"],
  info: ["info", "text-zinc-500 dark:text-zinc-400"],
};

function toast(title, tone = "info", detail = "") {
  const [iconName, color] = TOAST_TONES[tone];
  const el = h(
    "div",
    {
      class: "pointer-events-auto flex gap-3 rounded-xl border border-zinc-200 bg-white p-3.5 shadow-lg dark:border-zinc-800 dark:bg-zinc-900",
      role: tone === "error" ? "alert" : "status",
    },
    h("span", { class: `mt-0.5 ${color}` }, icon(iconName)),
    h(
      "div",
      { class: "min-w-0 flex-1" },
      h("p", { class: "text-sm font-medium" }, title),
      detail ? h("p", { class: "mt-0.5 break-words text-xs text-zinc-500 dark:text-zinc-400" }, detail) : null,
    ),
    h("button", { type: "button", "aria-label": "Dismiss", class: "self-start rounded p-0.5 text-zinc-400 hover:text-zinc-700 dark:hover:text-white", onclick: () => el.remove() }, icon("close", 14)),
  );
  $("#toasts").append(el);
  setTimeout(() => el.remove(), tone === "error" ? 10000 : 5000);
}

// ---------------------------------------------------------------- state

const state = {
  schema: null,
  categories: [], // tabs, from the schema's category enum
  prefixes: {}, // category -> id prefix for new rows (PRJ, FAQ, ...)
  columns: [], // grid columns, from the schema
  fields: [], // every writable field name
  rows: [], // { key, original: entry | null, data: {...}, errors: {field: message} }
  tab: null,
  syncDirty: false, // saved changes the chatbot hasn't indexed yet
  lastReport: null,
  syncing: false,
  sessionId: null,
};
let newRowCounter = 0;

// ---------------------------------------------------------------- auth

function showLogin(message = "") {
  $("#app").hidden = true;
  $("#login").hidden = false;
  closeTester();
  const error = $("#login-error");
  error.textContent = message;
  error.classList.toggle("hidden", !message);
  $("#password").value = "";
  $("#password").focus();
}

$("#login-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = $("#login-form button[type=submit]");
  button.disabled = true;
  try {
    await api(`${ADMIN}/auth`, { method: "POST", body: { password: $("#password").value } });
    state.schema = await api(`${ADMIN}/schema`);
    await enterApp();
  } catch (err) {
    showLogin(err.status === 429 ? "Too many attempts. Wait a minute and try again." : err.message);
  } finally {
    button.disabled = false;
  }
});

$("#logout-btn").addEventListener("click", async () => {
  if (dirtyRows().length && !confirm("You have unsaved changes. Sign out and lose them?")) return;
  await api(`${ADMIN}/logout`, { method: "POST" }).catch(() => {});
  state.rows = [];
  showLogin("Signed out.");
});

async function enterApp() {
  $("#login").hidden = true;
  $("#app").hidden = false;
  readSchema();
  await Promise.all([loadRows(), loadSyncStatus()]);
}

// ---------------------------------------------------------------- schema -> tabs and columns

const COLUMN_WIDTH = { order: "w-[92px]", id: "w-[124px]", text: "w-[230px]", url: "w-[230px]", boolean: "w-[78px]", textarea: "" };
const PATTERN_HINTS = { id: "Use 1–64 letters, digits, or . _ : -", url: "Must be empty or start with http:// or https://" };

function readSchema() {
  const props = state.schema.properties;
  const required = new Set(state.schema.required || []);
  const category = Object.entries(props).find(([, p]) => p.ui === "tabs");
  state.categories = category[1].enum;
  state.prefixes = category[1].id_prefixes || {};
  state.tab ??= state.categories[0];
  state.fields = Object.keys(props);
  state.columns = Object.entries(props)
    .filter(([, p]) => p.ui !== "tabs")
    .map(([name, p]) => ({
      ...p,
      name,
      required: required.has(name),
      widget: p.ui || { boolean: "boolean", integer: "order" }[p.type] || "text",
    }))
    .sort((a, b) => (b.widget === "order") - (a.widget === "order")); // order column first
}

// ---------------------------------------------------------------- rows

const pick = (entry) => Object.fromEntries(state.fields.map((f) => [f, entry[f] ?? (f === "url" || f === "title" ? "" : entry[f])]));
const isDirty = (row) => !row.original || state.fields.some((f) => row.data[f] !== (row.original[f] ?? (f === "url" || f === "title" ? "" : row.original[f])));
const dirtyRows = () => state.rows.filter(isDirty);

async function loadRows() {
  $("#kb-count").textContent = "Loading…";
  try {
    const entries = await api(`${ADMIN}/entries`);
    state.rows = entries.map((entry) => ({ key: entry.id, original: entry, data: pick(entry), errors: {} }));
  } catch (err) {
    state.rows = [];
    if (err.status !== 401) toast("Could not load the knowledge base", "error", err.message);
  }
  renderAll();
}

function nextId(category) {
  const prefix = state.prefixes[category] || category.replace(/[^A-Za-z]/g, "").slice(0, 3).toUpperCase();
  const pattern = new RegExp(`^${prefix}(\\d+)$`);
  const used = state.rows.map((r) => Number(r.data.id.match(pattern)?.[1] || 0));
  return prefix + String(Math.max(0, ...used) + 1).padStart(3, "0");
}

function tabRows(category = state.tab) {
  return state.rows
    .filter((r) => r.data.category === category)
    .sort((a, b) => a.data.sort_order - b.data.sort_order || a.data.id.localeCompare(b.data.id));
}

function addRow(values = {}) {
  const category = values.category || state.tab;
  const row = {
    key: `new-${++newRowCounter}`,
    original: null,
    errors: {},
    data: {
      id: nextId(category),
      title: "",
      content: "",
      url: "",
      active: true,
      sort_order: Math.max(0, ...tabRows(category).map((r) => r.data.sort_order)) + 1,
      ...values,
      category,
    },
  };
  state.rows.push(row);
  return row;
}

function moveRow(row, step) {
  const rows = tabRows();
  const from = rows.indexOf(row);
  const to = from + step;
  if (to < 0 || to >= rows.length) return;
  [rows[from], rows[to]] = [rows[to], rows[from]];
  rows.forEach((r, i) => (r.data.sort_order = i + 1)); // renumber the tab 1..n
  renderGrid();
  focusCell(row, "sort_order");
}

function validateRow(row) {
  const errors = {};
  for (const col of state.columns) {
    const value = row.data[col.name];
    if (typeof value !== "string") continue;
    if (col.required && !value.trim()) errors[col.name] = "Required.";
    else if (col.maxLength && value.length > col.maxLength) errors[col.name] = `At most ${col.maxLength.toLocaleString()} characters.`;
    else if (col.pattern && !new RegExp(col.pattern).test(value)) errors[col.name] = PATTERN_HINTS[col.widget] || "Invalid format.";
  }
  if (!errors.id && state.rows.some((other) => other !== row && other.data.id === row.data.id)) {
    errors.id = "This ID is already used.";
  }
  return errors;
}

// ---------------------------------------------------------------- rendering

function renderAll() {
  renderTabs();
  renderGrid();
}

const tabSignature = () =>
  state.categories.map((c) => `${tabRows(c).length}:${tabRows(c).some(isDirty)}`).join("|");

function renderTabs() {
  const dirtyByTab = new Set(dirtyRows().map((r) => r.data.category));
  state.renderedTabs = tabSignature();
  $("#tabs").replaceChildren(
    ...state.categories.map((category) => {
      const selected = category === state.tab;
      return h(
        "button",
        {
          type: "button",
          role: "tab",
          "aria-selected": String(selected),
          class:
            "inline-flex shrink-0 items-center gap-2 border-b-2 px-3 py-2.5 text-sm font-medium transition " +
            (selected
              ? "border-zinc-900 text-zinc-900 dark:border-white dark:text-white"
              : "border-transparent text-zinc-500 hover:border-zinc-300 hover:text-zinc-800 dark:text-zinc-400 dark:hover:border-zinc-600 dark:hover:text-zinc-200"),
          onclick: () => {
            state.tab = category;
            $("#search").value = "";
            renderAll();
          },
        },
        label(category),
        h(
          "span",
          { class: `rounded-full px-1.5 py-0.5 text-[11px] tabular-nums ${selected ? "bg-zinc-900 text-white dark:bg-white dark:text-zinc-900" : "bg-zinc-100 text-zinc-600 dark:bg-zinc-800 dark:text-zinc-300"}` },
          tabRows(category).length,
        ),
        dirtyByTab.has(category) ? h("span", { class: "h-1.5 w-1.5 rounded-full bg-amber-500", title: "Unsaved changes" }) : null,
      );
    }),
  );
}

const CELL = "border-b border-zinc-100 px-1 py-1 align-top dark:border-zinc-800/80";
const INPUT =
  "block w-full rounded-md border border-transparent bg-transparent px-2 py-1.5 text-sm outline-none transition placeholder:text-zinc-400 " +
  "hover:border-zinc-200 focus:border-zinc-400 focus:bg-white focus:ring-2 focus:ring-zinc-900/10 aria-[invalid=true]:border-red-500 aria-[invalid=true]:bg-red-50/60 " +
  "read-only:cursor-default read-only:text-zinc-500 read-only:hover:border-transparent dark:hover:border-zinc-700 dark:focus:border-zinc-500 dark:focus:bg-zinc-950 " +
  "dark:focus:ring-white/10 dark:aria-[invalid=true]:bg-red-950/30";

function renderGrid() {
  $("#grid-head").replaceChildren(
    h(
      "tr",
      {},
      state.columns.map((col) =>
        h(
          "th",
          { class: `border-b border-zinc-200 px-3 py-2.5 font-medium dark:border-zinc-800 ${COLUMN_WIDTH[col.widget] ?? ""}`, title: col.description || "", scope: "col" },
          col.title || col.name,
          col.required ? h("span", { class: "ml-0.5 text-red-500", "aria-hidden": "true" }, "*") : null,
        ),
      ),
      h("th", { class: "w-[96px] border-b border-zinc-200 px-3 py-2.5 font-medium dark:border-zinc-800", scope: "col" }, "Updated"),
      h("th", { class: "w-[48px] border-b border-zinc-200 dark:border-zinc-800", scope: "col" }, h("span", { class: "sr-only" }, "Actions")),
    ),
  );

  const query = $("#search").value.trim().toLowerCase();
  const all = tabRows();
  const rows = query ? all.filter((r) => state.fields.some((f) => String(r.data[f]).toLowerCase().includes(query))) : all;
  $("#grid-body").replaceChildren(...rows.map((row) => rowEl(row, all, Boolean(query))));
  requestAnimationFrame(() => document.querySelectorAll("#grid-body textarea").forEach(autosize));

  $("#empty").hidden = rows.length > 0;
  $("#empty-title").textContent = query ? "No matching rows" : `No ${label(state.tab)} yet`;
  $("#empty-text").textContent = query
    ? "Try a different search."
    : "Click Add row, paste rows copied from a spreadsheet, or use Bulk JSON import (try data/sample_entries.json).";
  updateCounters();
}

function rowEl(row, tabList, filtered) {
  const dirty = isDirty(row);
  const hasErrors = Object.keys(row.errors).length > 0;
  const bar = hasErrors ? "shadow-[inset_3px_0_0_0_rgb(239_68_68)]" : !row.original ? "shadow-[inset_3px_0_0_0_rgb(16_185_129)]" : dirty ? "shadow-[inset_3px_0_0_0_rgb(245_158_11)]" : "";
  const tr = h("tr", { class: `group transition ${dirty ? "bg-amber-50/40 dark:bg-amber-950/10" : "hover:bg-zinc-50/70 dark:hover:bg-zinc-800/30"}`, "data-key": row.key });
  state.columns.forEach((col, i) => tr.append(h("td", { class: `${CELL} ${i === 0 ? bar : ""}` }, cellEl(row, col, tabList, filtered))));
  tr.append(
    h(
      "td",
      { class: `${CELL} px-3 py-2.5 text-xs text-zinc-400` },
      !row.original
        ? h("span", { class: "rounded bg-emerald-100 px-1.5 py-0.5 font-medium text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300" }, "New")
        : dirty
          ? h("span", { class: "rounded bg-amber-100 px-1.5 py-0.5 font-medium text-amber-800 dark:bg-amber-950 dark:text-amber-300" }, "Edited")
          : row.original.updated_at
            ? ago(row.original.updated_at)
            : "",
    ),
    h(
      "td",
      { class: `${CELL} text-center` },
      h(
        "button",
        {
          type: "button",
          title: row.original ? "Delete row" : "Remove unsaved row",
          "aria-label": `Delete ${row.data.id}`,
          class: "mt-0.5 rounded-lg p-1.5 text-zinc-400 opacity-60 transition hover:bg-red-50 hover:text-red-600 group-hover:opacity-100 focus-visible:opacity-100 dark:hover:bg-red-950/50 dark:hover:text-red-400",
          onclick: () => deleteRow(row),
        },
        icon("trash"),
      ),
    ),
  );
  return tr;
}

function cellEl(row, col, tabList, filtered) {
  const value = row.data[col.name];
  const error = row.errors[col.name];
  const common = {
    "data-field": col.name,
    "aria-label": `${col.title || col.name} for ${row.data.id}`,
    "aria-invalid": error ? "true" : "false",
    title: error || "",
  };
  const onEdit = (newValue) => {
    row.data[col.name] = newValue;
    if (row.errors[col.name]) {
      delete row.errors[col.name];
      const el = $(`tr[data-key="${CSS.escape(row.key)}"] [data-field="${col.name}"]`);
      el?.setAttribute("aria-invalid", "false");
      el?.removeAttribute("title");
      el?.parentElement.querySelector(".cell-error")?.remove();
    }
    refreshRowState(row);
  };

  let control;
  if (col.widget === "order") {
    const position = tabList.indexOf(row);
    control = h(
      "div",
      { class: "flex items-center gap-1 px-1 py-1" },
      h("span", { class: "w-7 text-right font-mono text-xs tabular-nums text-zinc-400", "data-field": col.name }, value),
      h(
        "div",
        { class: "flex flex-col" },
        h("button", { type: "button", "aria-label": `Move ${row.data.id} up`, disabled: filtered || position === 0, class: "rounded p-0.5 text-zinc-400 hover:bg-zinc-100 hover:text-zinc-900 disabled:opacity-25 dark:hover:bg-zinc-800 dark:hover:text-white", onclick: () => moveRow(row, -1) }, icon("up", 14)),
        h("button", { type: "button", "aria-label": `Move ${row.data.id} down`, disabled: filtered || position === tabList.length - 1, class: "rounded p-0.5 text-zinc-400 hover:bg-zinc-100 hover:text-zinc-900 disabled:opacity-25 dark:hover:bg-zinc-800 dark:hover:text-white", onclick: () => moveRow(row, 1) }, icon("down", 14)),
      ),
    );
  } else if (col.widget === "boolean") {
    const input = h("input", { ...common, type: "checkbox", role: "switch", checked: Boolean(value), class: "peer sr-only", onchange: (e) => onEdit(e.target.checked) });
    control = h(
      "label",
      { class: "relative mx-2 mt-1.5 inline-flex h-5 w-9 cursor-pointer items-center" },
      input,
      h("span", { class: "absolute inset-0 rounded-full bg-zinc-300 transition peer-checked:bg-emerald-600 peer-focus-visible:ring-2 peer-focus-visible:ring-emerald-500 peer-focus-visible:ring-offset-1 dark:bg-zinc-700" }),
      h("span", { class: "absolute left-0.5 h-4 w-4 rounded-full bg-white shadow transition peer-checked:translate-x-4" }),
    );
  } else if (col.widget === "textarea" || col.widget === "text") {
    // "text" cells are single-line values that wrap so long titles stay readable.
    const singleLine = col.widget === "text";
    control = h("textarea", {
      ...common,
      rows: 1,
      value: value ?? "",
      maxLength: col.maxLength,
      placeholder: singleLine ? "" : "Write the fact in full sentences…",
      class: `${INPUT} min-h-[2.25rem] resize-none leading-relaxed ${singleLine ? "font-medium" : ""}`,
      onkeydown: (e) => singleLine && e.key === "Enter" && e.preventDefault(),
      oninput: (e) => {
        if (singleLine && /[\r\n]/.test(e.target.value)) e.target.value = e.target.value.replace(/[\r\n]+/g, " ");
        autosize(e.target);
        onEdit(e.target.value);
      },
    });
  } else {
    control = h("input", {
      ...common,
      type: col.widget === "url" ? "url" : "text",
      value: value ?? "",
      maxLength: col.maxLength,
      readOnly: col.widget === "id" && Boolean(row.original), // a saved row's id is its identity
      placeholder: col.widget === "url" ? "https://" : "",
      spellcheck: col.widget !== "id" && col.widget !== "url",
      class: `${INPUT} ${col.widget === "id" ? "font-mono text-xs" : ""}`,
      oninput: (e) => onEdit(e.target.value),
    });
  }
  return h("div", {}, control, error ? h("p", { class: "cell-error px-2 pt-0.5 text-[11px] font-medium text-red-600 dark:text-red-400" }, error) : null);
}

function autosize(area) {
  area.style.height = "auto";
  area.style.height = `${Math.min(area.scrollHeight + 2, 320)}px`;
}

function refreshRowState(row) {
  // Restyle one row without re-rendering it, so the focused cell keeps focus.
  const tr = $(`tr[data-key="${CSS.escape(row.key)}"]`);
  if (tr) {
    const fresh = rowEl(row, tabRows(), Boolean($("#search").value.trim()));
    tr.className = fresh.className;
    tr.firstElementChild.className = fresh.firstElementChild.className;
    tr.children[tr.children.length - 2].replaceChildren(...fresh.children[fresh.children.length - 2].childNodes);
  }
  updateCounters();
}

function updateCounters() {
  const dirty = dirtyRows().length;
  $("#dirty-count").textContent = dirty ? `${dirty} unsaved change${dirty === 1 ? "" : "s"}` : "";
  $("#save-btn").disabled = dirty === 0;
  $("#discard-btn").hidden = dirty === 0;
  const active = state.rows.filter((r) => r.original?.active).length;
  const saved = state.rows.filter((r) => r.original).length;
  $("#kb-count").textContent = `${saved} rows in Supabase · ${active} active`;
  if (tabSignature() !== state.renderedTabs) renderTabs(); // counts or unsaved dots changed
  renderBadge();
}

function focusCell(row, field) {
  const el = $(`tr[data-key="${CSS.escape(row.key)}"] [data-field="${field}"]`);
  if (el?.tagName === "SPAN") el.closest("td").querySelector("button:not(:disabled)")?.focus();
  else el?.focus();
  el?.scrollIntoView({ block: "nearest" });
}

// ---------------------------------------------------------------- editing actions

$("#add-row-btn").addEventListener("click", () => {
  $("#search").value = "";
  const row = addRow();
  renderAll();
  focusCell(row, "title");
});

async function deleteRow(row) {
  if (!row.original) {
    state.rows = state.rows.filter((r) => r !== row);
    renderAll();
    return;
  }
  const dialog = $("#confirm-dialog");
  $("#confirm-text").textContent = `“${row.data.title || row.data.id}” (${row.data.id}) will be deleted from Supabase now. Sync afterwards to remove it from the chatbot.`;
  dialog.returnValue = "";
  dialog.addEventListener(
    "close",
    async () => {
      if (dialog.returnValue !== "confirm") return;
      try {
        await api(`${ADMIN}/entries/${encodeURIComponent(row.original.id)}`, { method: "DELETE" });
        state.rows = state.rows.filter((r) => r !== row);
        markSyncDirty();
        renderAll();
        toast(`Deleted ${row.data.id}`, "success", "Sync to Qdrant to remove it from the chatbot.");
      } catch (err) {
        if (err.status !== 401) toast("Could not delete the row", "error", err.message);
      }
    },
    { once: true },
  );
  dialog.showModal();
}

async function saveAll() {
  const dirty = dirtyRows();
  if (!dirty.length) return true;
  let firstBad = null;
  for (const row of dirty) {
    row.errors = validateRow(row);
    if (!firstBad && Object.keys(row.errors).length) firstBad = row;
  }
  if (firstBad) {
    showFirstError(firstBad);
    toast("Fix the highlighted cells before saving", "error");
    return false;
  }

  const button = $("#save-btn");
  button.disabled = true;
  try {
    const result = await api(`${ADMIN}/entries/bulk`, { method: "POST", body: dirty.map((r) => r.data) });
    markSyncDirty();
    await loadRows();
    toast(`Saved ${result.upserted} row${result.upserted === 1 ? "" : "s"}`, "success", "Sync to Qdrant to update the chatbot.");
    return true;
  } catch (err) {
    if (err.status === 422) {
      for (const detail of err.details) {
        const [index, field] = detail.field.split(".");
        const row = dirty[Number(index)];
        if (row) row.errors[field || "id"] = detail.message;
      }
      const bad = dirty.find((r) => Object.keys(r.errors).length);
      if (bad) showFirstError(bad);
    }
    if (err.status !== 401) toast("Could not save your changes", "error", err.message);
    updateCounters();
    return false;
  }
}

function showFirstError(row) {
  state.tab = row.data.category;
  $("#search").value = "";
  renderAll();
  focusCell(row, Object.keys(row.errors)[0]);
}

$("#save-btn").addEventListener("click", saveAll);
$("#discard-btn").addEventListener("click", () => {
  if (!confirm("Discard all unsaved changes?")) return;
  state.rows = state.rows.filter((r) => r.original).map((r) => ({ ...r, data: pick(r.original), errors: {} }));
  renderAll();
  toast("Changes discarded");
});
$("#reload-btn").addEventListener("click", () => {
  if (dirtyRows().length && !confirm("Reload from Supabase and lose your unsaved changes?")) return;
  loadRows();
});
$("#search").addEventListener("input", renderGrid);
window.addEventListener("beforeunload", (event) => {
  if (dirtyRows().length) {
    event.preventDefault();
    event.returnValue = "";
  }
});

// ---------------------------------------------------------------- paste rows from a spreadsheet

function parseTsv(text) {
  // Google Sheets / Excel copy: tab-separated, multi-line cells wrapped in quotes ("" = ").
  const rows = [];
  let row = [];
  let cell = "";
  let quoted = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') {
        cell += '"';
        i++;
      } else if (ch === '"') quoted = false;
      else cell += ch;
    } else if (ch === '"' && cell === "") quoted = true;
    else if (ch === "\t") {
      row.push(cell);
      cell = "";
    } else if (ch === "\n" || ch === "\r") {
      if (ch === "\r" && text[i + 1] === "\n") i++;
      row.push(cell);
      rows.push(row);
      row = [];
      cell = "";
    } else cell += ch;
  }
  if (cell || row.length) {
    row.push(cell);
    rows.push(row);
  }
  return rows.filter((r) => r.some((c) => c.trim()));
}

$("#grid-wrap").addEventListener("paste", (event) => {
  const text = event.clipboardData?.getData("text/plain") || "";
  if (!text.includes("\t")) return; // a normal paste into one cell
  event.preventDefault();
  const table = parseTsv(text);
  const first = table[0].map((c) => c.trim().toLowerCase());
  const columns = first.includes("content") ? (table.shift(), first) : ["id", "title", "content", "url", "active", "sort_order"];
  let created = 0;
  let updated = 0;
  const touchedTabs = new Set();
  for (const cells of table) {
    const values = {};
    columns.forEach((name, i) => {
      if (state.fields.includes(name) && cells[i] !== undefined && cells[i].trim() !== "") values[name] = cells[i].trim();
    });
    if ("active" in values) values.active = !/^(false|0|no|n|f|inactive)$/i.test(values.active);
    if ("sort_order" in values) values.sort_order = Number.parseInt(values.sort_order, 10) || 0;
    if (values.category && !state.categories.includes(values.category)) delete values.category;
    const existing = values.id && state.rows.find((r) => r.data.id === values.id);
    if (existing) {
      Object.assign(existing.data, values, { id: existing.data.id });
      touchedTabs.add(existing.data.category);
      updated++;
    } else {
      touchedTabs.add(addRow(values).data.category);
      created++;
    }
  }
  $("#search").value = "";
  renderAll();
  const elsewhere = [...touchedTabs].filter((t) => t !== state.tab).map(label);
  toast(
    `Pasted ${created} new and ${updated} updated row${created + updated === 1 ? "" : "s"}`,
    "success",
    `Review them, then Save changes.${elsewhere.length ? ` Also changed: ${elsewhere.join(", ")}.` : ""}`,
  );
});

// ---------------------------------------------------------------- bulk JSON import

function parseBulk() {
  const text = $("#bulk-text").value.trim();
  if (!text) return { entries: null, error: "" };
  try {
    let data = JSON.parse(text);
    if (data && !Array.isArray(data)) data = data.records ?? data.entries;
    if (!Array.isArray(data)) return { error: 'Expected a JSON array of rows (or an object with a "records" array).' };
    if (data.length > 2000) return { error: `${data.length} rows — import at most 2,000 at a time.` };
    return { entries: data };
  } catch (err) {
    return { error: `Invalid JSON: ${err.message}` };
  }
}

function updateBulkSummary() {
  const { entries, error } = parseBulk();
  const summary = $("#bulk-summary");
  summary.textContent = error || (entries ? `${entries.length} row${entries.length === 1 ? "" : "s"} ready to import` : "");
  summary.className = `text-xs ${error ? "text-red-600 dark:text-red-400" : "text-zinc-500 dark:text-zinc-400"}`;
  $("#bulk-submit").disabled = !entries || !entries.length;
  $("#bulk-errors").hidden = true;
}

$("#bulk-btn").addEventListener("click", () => {
  if (dirtyRows().length) return toast("Save or discard your grid changes first", "info", "A bulk import reloads the grid from Supabase.");
  updateBulkSummary();
  $("#bulk-dialog").showModal();
  $("#bulk-text").focus();
});
$("#bulk-text").addEventListener("input", updateBulkSummary);
$("#bulk-file").addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  $("#bulk-text").value = await file.text();
  event.target.value = "";
  updateBulkSummary();
});

$("#bulk-submit").addEventListener("click", async () => {
  const { entries } = parseBulk();
  if (!entries) return;
  const button = $("#bulk-submit");
  button.disabled = true;
  try {
    const result = await api(`${ADMIN}/entries/bulk`, { method: "POST", body: entries });
    $("#bulk-dialog").close();
    $("#bulk-text").value = "";
    const merged = result.duplicates_merged ? ` (${result.duplicates_merged} duplicate IDs merged)` : "";
    toast(`Imported ${result.upserted} rows${merged}`, "success", "Sync to Qdrant to make them searchable.");
    markSyncDirty();
    await loadRows();
  } catch (err) {
    const list = $("#bulk-errors");
    const lines =
      err.status === 422
        ? err.details.slice(0, 25).map((d) => {
            const [index, ...field] = d.field.split(".");
            return /^\d+$/.test(index) ? `Row ${Number(index) + 1} · ${field.join(".") || "row"}: ${d.message}` : `${d.field}: ${d.message}`;
          })
        : [err.message];
    list.replaceChildren(...lines.map((line) => h("li", {}, line)));
    list.hidden = false;
  } finally {
    button.disabled = false;
  }
});

// ---------------------------------------------------------------- sync

const STAGES = ["fetching", "embedding", "upserting"];
const STEP_DOT = {
  pending: "h-5 w-5 rounded-full border-2 border-zinc-300 dark:border-zinc-700",
  active: "h-5 w-5 animate-spin rounded-full border-2 border-emerald-500 border-t-transparent",
  done: "grid h-5 w-5 place-items-center rounded-full bg-emerald-600 text-white",
  failed: "grid h-5 w-5 place-items-center rounded-full bg-red-600 text-white",
};
const STEP_TEXT = {
  pending: "text-zinc-400 dark:text-zinc-500",
  active: "font-medium text-zinc-900 dark:text-white",
  done: "text-zinc-600 dark:text-zinc-300",
  failed: "font-medium text-red-600 dark:text-red-400",
};

function setStage(stage, outcome) {
  const current = STAGES.indexOf(stage);
  for (const step of document.querySelectorAll("#sync-progress [data-step]")) {
    const i = STAGES.indexOf(step.dataset.step);
    const status = outcome === "done" || i < current ? "done" : i > current ? "pending" : outcome === "failed" ? "failed" : "active";
    step.className = `flex items-center gap-2 ${STEP_TEXT[status]}`;
    const dot = step.querySelector(".step-dot");
    dot.className = `step-dot ${STEP_DOT[status]}`;
    dot.replaceChildren(status === "done" ? icon("check", 12) : status === "failed" ? icon("close", 12) : "");
  }
}

const BADGE = {
  emerald: ["border-emerald-200 bg-emerald-50 text-emerald-800 dark:border-emerald-900 dark:bg-emerald-950/50 dark:text-emerald-300", "bg-emerald-500"],
  amber: ["border-amber-200 bg-amber-50 text-amber-800 dark:border-amber-900 dark:bg-amber-950/50 dark:text-amber-300", "bg-amber-500"],
  red: ["border-red-200 bg-red-50 text-red-800 dark:border-red-900 dark:bg-red-950/50 dark:text-red-300", "bg-red-500"],
  zinc: ["border-zinc-200 bg-zinc-50 text-zinc-600 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-400", "bg-zinc-400"],
};

function renderBadge() {
  const report = state.lastReport;
  let tone = "zinc";
  let text = "Not synced since server start";
  let title = "";
  if (dirtyRows().length) {
    tone = "amber";
    text = "Unsaved changes";
    title = "Sync saves them first.";
  } else if (state.syncDirty) {
    tone = "amber";
    text = "Not yet synced";
    title = "Supabase has changes the chatbot hasn't indexed yet.";
  } else if (report && !report.success) {
    tone = "red";
    text = "Last sync failed";
    title = report.error || "";
  } else if (report) {
    tone = "emerald";
    text = `Synced ${ago(report.finished_at)} · ${report.total_entries} rows`;
    title = `${report.vectors_upserted} vectors upserted · ${report.unchanged} unchanged · ${report.removed} removed · ${fmtMs(report.duration_ms)}`;
  }
  const [classes, dot] = BADGE[tone];
  const badge = $("#sync-badge");
  badge.className = `hidden items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs sm:inline-flex ${classes}`;
  badge.title = title;
  badge.replaceChildren(h("span", { class: `h-1.5 w-1.5 rounded-full ${dot}` }), text);
}

function markSyncDirty() {
  state.syncDirty = true;
  renderBadge();
}

async function loadSyncStatus() {
  try {
    const status = await api(`${ADMIN}/sync`);
    state.lastReport = status.last_report;
    const last = state.lastReport;
    state.syncDirty = Boolean(
      last?.success && state.rows.some((r) => r.original?.updated_at && new Date(r.original.updated_at) > new Date(last.finished_at)),
    );
  } catch {
    state.lastReport = null;
  }
  renderBadge();
}

async function runSync() {
  if (state.syncing) return;
  if (dirtyRows().length && !(await saveAll())) return; // sync reads Supabase: save the grid first
  state.syncing = true;
  const button = $("#sync-btn");
  button.disabled = true;
  button.lastElementChild.textContent = "Syncing…";
  $("#sync-progress").hidden = false;
  let stage = "fetching";
  setStage(stage);
  const poll = setInterval(async () => {
    try {
      const status = await api(`${ADMIN}/sync`);
      if (status.running && STAGES.includes(status.stage) && state.syncing) {
        stage = status.stage;
        setStage(stage);
      }
    } catch {}
  }, 400);
  try {
    const report = await api(`${ADMIN}/sync`, { method: "POST" });
    clearInterval(poll);
    state.lastReport = report;
    if (report.success) {
      state.syncDirty = false;
      setStage(stage, "done");
      toast(
        `Synced ${report.total_entries} rows in ${fmtMs(report.duration_ms)}`,
        "success",
        `${report.chunks_embedded} chunks embedded · ${report.vectors_upserted} vectors upserted · ${report.removed} removed · ${report.unchanged} unchanged`,
      );
    } else {
      setStage(stage, "failed");
      toast("Sync failed", "error", report.error || "Unknown error.");
    }
  } catch (err) {
    clearInterval(poll);
    setStage(stage, "failed");
    if (err.status !== 401) toast("Sync failed", "error", err.message);
  } finally {
    state.syncing = false;
    button.disabled = false;
    button.lastElementChild.textContent = "Sync to Qdrant";
    renderBadge();
    setTimeout(() => {
      if (!state.syncing) $("#sync-progress").hidden = true;
    }, 4000);
  }
}

$("#sync-btn").addEventListener("click", runSync);
setInterval(() => !$("#app").hidden && renderBadge(), 30000);

// ---------------------------------------------------------------- chat tester

$("#tester-endpoint").textContent = `${API}/chat`;

function openTester() {
  const panel = $("#tester");
  panel.classList.remove("translate-x-full");
  panel.removeAttribute("inert");
  panel.setAttribute("aria-hidden", "false");
  $("#tester-backdrop").hidden = false;
  $("#tester-input").focus();
}

function closeTester() {
  const panel = $("#tester");
  panel.classList.add("translate-x-full");
  panel.setAttribute("inert", "");
  panel.setAttribute("aria-hidden", "true");
  $("#tester-backdrop").hidden = true;
}

function bubble(role, ...children) {
  const mine = role === "user";
  return h(
    "div",
    { class: `flex ${mine ? "justify-end" : "justify-start"}` },
    h(
      "div",
      {
        class: mine
          ? "max-w-[85%] rounded-2xl rounded-br-md bg-zinc-900 px-3.5 py-2 text-sm text-white dark:bg-white dark:text-zinc-900"
          : role === "error"
            ? "max-w-[92%] rounded-2xl rounded-bl-md border border-red-200 bg-red-50 px-3.5 py-2.5 text-sm text-red-700 dark:border-red-900 dark:bg-red-950/50 dark:text-red-300"
            : "w-full max-w-[92%] rounded-2xl rounded-bl-md border border-zinc-200 bg-zinc-50 px-3.5 py-2.5 dark:border-zinc-800 dark:bg-zinc-950",
      },
      ...children,
    ),
  );
}

function replyText(text) {
  const paragraph = h("p", { class: "whitespace-pre-wrap text-sm leading-relaxed" });
  for (const part of text.split(/(\[\d+\])/)) {
    const marker = part.match(/^\[(\d+)\]$/);
    paragraph.append(
      marker
        ? h("sup", { class: "mx-0.5 rounded bg-zinc-200 px-1 font-mono text-[10px] text-zinc-700 dark:bg-zinc-700 dark:text-zinc-200" }, marker[1])
        : part,
    );
  }
  return paragraph;
}

function sourceCard(source, n) {
  const url = source.url ? safeUrl(source.url) : null;
  return h(
    "li",
    { class: "rounded-lg border border-zinc-200 bg-white p-2.5 dark:border-zinc-800 dark:bg-zinc-900" },
    h(
      "div",
      { class: "flex items-center gap-1.5 text-xs" },
      h("span", { class: "font-mono text-zinc-400" }, `[${n}]`),
      h("span", { class: "truncate font-medium" }, source.title),
      h("span", { class: "rounded bg-zinc-100 px-1.5 py-0.5 text-[10px] text-zinc-600 dark:bg-zinc-800 dark:text-zinc-300" }, label(source.category)),
      source.score != null ? h("span", { class: "ml-auto font-mono text-[10px] text-zinc-400", title: "Relevance score" }, source.score.toFixed(3)) : null,
    ),
    h("p", { class: "mt-1.5 line-clamp-4 text-xs leading-relaxed text-zinc-600 dark:text-zinc-400" }, source.snippet),
    url
      ? h("a", { href: url, target: "_blank", rel: "noopener noreferrer", class: "mt-1.5 inline-flex items-center gap-1 text-[11px] text-emerald-700 hover:underline dark:text-emerald-400" }, icon("link", 11), new URL(url).hostname)
      : null,
  );
}

$("#tester-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const input = $("#tester-input");
  const message = input.value.trim();
  if (!message) return;
  const log = $("#tester-log");
  $("#tester-hint")?.remove();
  input.value = "";
  log.append(bubble("user", h("p", { class: "whitespace-pre-wrap" }, message)));
  const pending = bubble("assistant", h("p", { class: "animate-pulse text-sm text-zinc-400" }, "Retrieving and generating…"));
  log.append(pending);
  log.scrollTop = log.scrollHeight;

  const button = $("#tester-form button[type=submit]");
  button.disabled = true;
  const started = performance.now();
  try {
    const data = await api(`${API}/chat`, { method: "POST", body: { message, session_id: state.sessionId } });
    state.sessionId = data.session_id;
    const meta = h(
      "p",
      { class: "mt-2 text-[11px] text-zinc-400" },
      `${fmtMs(data.latency_ms)} server · ${fmtMs(performance.now() - started)} round trip · ${data.sources.length} source${data.sources.length === 1 ? "" : "s"}`,
    );
    const sources = data.sources.length
      ? h(
          "details",
          { class: "mt-2", open: true },
          h("summary", { class: "cursor-pointer select-none text-xs font-medium text-zinc-500 dark:text-zinc-400" }, "Retrieved source chunks"),
          h("ol", { class: "mt-2 space-y-2" }, data.sources.map((s, i) => sourceCard(s, i + 1))),
        )
      : null;
    pending.replaceWith(bubble("assistant", replyText(data.reply), meta, sources));
  } catch (err) {
    const retry = err.status === 429 || err.status === 503 ? " Try again shortly." : "";
    pending.replaceWith(bubble("error", h("p", {}, `${err.message}${retry}`)));
  } finally {
    button.disabled = false;
    log.scrollTop = log.scrollHeight;
    input.focus();
  }
});

$("#tester-input").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    $("#tester-form").requestSubmit();
  }
});
$("#tester-btn").addEventListener("click", openTester);
$("#tester-close").addEventListener("click", closeTester);
$("#tester-backdrop").addEventListener("click", closeTester);
$("#tester-reset").addEventListener("click", () => {
  state.sessionId = null;
  $("#tester-log").replaceChildren(h("p", { id: "tester-hint", class: "rounded-lg border border-dashed border-zinc-300 px-3 py-6 text-center text-sm text-zinc-500 dark:border-zinc-700 dark:text-zinc-400" }, "New session started."));
});

// ---------------------------------------------------------------- theme & shortcuts

$("#theme-btn").addEventListener("click", () => {
  const dark = document.documentElement.classList.toggle("dark");
  try {
    localStorage.setItem("theme", dark ? "dark" : "light");
  } catch {}
});

document.addEventListener("keydown", (event) => {
  if ($("#app").hidden) return;
  const typing = ["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName);
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") {
    event.preventDefault();
    if (!$("#bulk-dialog").open) saveAll();
  } else if (event.key === "/" && !typing) {
    event.preventDefault();
    $("#search").focus();
  } else if (event.key === "Escape" && !$("#tester").hasAttribute("inert")) {
    closeTester();
  }
});

// ---------------------------------------------------------------- boot

(async function boot() {
  try {
    state.schema = await api(`${ADMIN}/schema`);
  } catch (err) {
    return showLogin(err.status === 401 ? "" : err.message);
  }
  await enterApp();
})();
