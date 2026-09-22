// tbkit's docs are light-mode only (no theme-switcher in the navbar).
// pydata_sphinx_theme stores the light/dark preference in localStorage,
// which is shared across every cpoli.github.io/<project>/ site -- so a
// dark preference set on a sibling project's docs would otherwise carry
// over here. Force it back to light on every load.
localStorage.setItem("mode", "light");
localStorage.setItem("theme", "light");
document.documentElement.dataset.mode = "light";
document.documentElement.dataset.theme = "light";
