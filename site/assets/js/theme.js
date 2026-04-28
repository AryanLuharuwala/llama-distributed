/* theme.js — dark/light theme toggle (initial paint is set inline in <head>) */
(function () {
  var KEY = 'ld-theme';
  var root = document.documentElement;
  var transitionTimer = null;

  function current() {
    return root.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
  }

  function apply(theme, animate) {
    if (animate) {
      // Brief class that enables a global cross-fade between palettes
      root.classList.add('theme-transitioning');
      if (transitionTimer) clearTimeout(transitionTimer);
      transitionTimer = setTimeout(function () {
        root.classList.remove('theme-transitioning');
      }, 420);
    }
    if (theme === 'dark') root.setAttribute('data-theme', 'dark');
    else root.removeAttribute('data-theme');
  }

  function wire() {
    var buttons = document.querySelectorAll('[data-theme-toggle]');
    for (var i = 0; i < buttons.length; i++) {
      (function (btn) {
        if (btn.__themeBound) return;
        btn.__themeBound = true;
        btn.addEventListener('click', function () {
          var next = current() === 'dark' ? 'light' : 'dark';
          apply(next, true);
          btn.setAttribute('aria-pressed', next === 'dark' ? 'true' : 'false');
          try { localStorage.setItem(KEY, next); } catch (e) {}
        });
        // Reflect initial state for screen readers
        btn.setAttribute('aria-pressed', current() === 'dark' ? 'true' : 'false');
      })(buttons[i]);
    }
  }

  // DOM may already be parsed (script is at end of <body>) — wire either way
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', wire);
  } else {
    wire();
  }

  // Track system preference if the user has not made an explicit choice yet
  if (window.matchMedia) {
    var mq = window.matchMedia('(prefers-color-scheme: dark)');
    var handler = function (ev) {
      try { if (localStorage.getItem(KEY)) return; } catch (e) {}
      apply(ev.matches ? 'dark' : 'light', true);
      var buttons = document.querySelectorAll('[data-theme-toggle]');
      for (var i = 0; i < buttons.length; i++) {
        buttons[i].setAttribute('aria-pressed', ev.matches ? 'true' : 'false');
      }
    };
    if (mq.addEventListener) mq.addEventListener('change', handler);
    else if (mq.addListener) mq.addListener(handler);
  }
})();
