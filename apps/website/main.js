document.querySelectorAll("[data-copy]").forEach((btn) => {
  btn.addEventListener("click", async () => {
    const sel = btn.getAttribute("data-copy");
    const el = document.querySelector(sel);
    if (!el) return;
    try {
      await navigator.clipboard.writeText(el.textContent.trim());
      btn.textContent = "Copied!";
      setTimeout(() => (btn.textContent = "Copy"), 1200);
    } catch {
      btn.textContent = "Copy manually";
    }
  });
});

document.querySelectorAll(".tab").forEach((t) => {
  t.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
    document.querySelectorAll(".pane").forEach((x) => x.classList.remove("active"));
    t.classList.add("active");
    document.getElementById(t.dataset.tab)?.classList.add("active");
  });
});
