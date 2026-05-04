/* nav.js — mobile nav toggle + active link highlighting */

(function () {
  "use strict";

  document.addEventListener("DOMContentLoaded", function () {
    var toggle = /** @type {HTMLElement | null} */ (document.querySelector("[data-nav-toggle]"));
    var links  = /** @type {HTMLElement | null} */ (document.querySelector("[data-nav-links]"));
    if (toggle && links) {
      /** @type {HTMLElement} */
      var linksEl = links;
      /** @type {HTMLElement} */
      var toggleEl = toggle;
      toggleEl.addEventListener("click", function () {
        var open = linksEl.classList.toggle("open");
        toggleEl.setAttribute("aria-expanded", open ? "true" : "false");
      });
    }

    // Map the current path onto the closest-matching nav link.
    // Strategy: compute the last path segment of each nav link, then pick
    // the nav link whose segment appears in the current URL.
    var currentPath = location.pathname.replace(/\/$/, "") || "/";

    /**
     * @param {string} href
     * @returns {string}
     */
    function lastSegment(href) {
      if (!href) return "";
      // strip fragment + query
      href = href.split("#")[0].split("?")[0];
      // strip trailing slash
      href = href.replace(/\/$/, "");
      // collapse ../ prefixes
      var parts = href.split("/").filter(function (p) { return p && p !== ".."; });
      return parts[parts.length - 1] || "";
    }

    document.querySelectorAll("[data-nav-links] a").forEach(function (el) {
      var a = /** @type {HTMLAnchorElement} */ (el);
      var seg = lastSegment(a.getAttribute("href") || "");
      if (!seg) return;
      // Match either exact file (about.html) or directory marker (docs, blog).
      if (currentPath.endsWith("/" + seg) ||
          currentPath.indexOf("/" + seg + "/") !== -1 ||
          currentPath.endsWith("/" + seg + ".html")) {
        a.classList.add("active");
      }
    });

    document.querySelectorAll(".docs-side a").forEach(function (el) {
      var a = /** @type {HTMLAnchorElement} */ (el);
      var href = a.getAttribute("href") || "";
      var fileOnly = href.split("#")[0].replace(/^\.\//, "").replace(/^\.\.\//, "");
      if (fileOnly && location.pathname.endsWith("/" + fileOnly)) {
        a.classList.add("active");
      }
    });
  });
})();
