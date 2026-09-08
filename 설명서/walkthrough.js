(() => {
  const panels = [...document.querySelectorAll('.step')];
  const links = [...document.querySelectorAll('.steps a')];
  const previous = document.getElementById('previous');
  const next = document.getElementById('next');
  const position = document.getElementById('position');
  let current = 0;
  function show(index, focus = false) {
    current = Math.max(0, Math.min(index, panels.length - 1));
    panels.forEach((panel, i) => { panel.hidden = i !== current; });
    links.forEach((link, i) => {
      if (i === current) link.setAttribute('aria-current', 'step');
      else link.removeAttribute('aria-current');
    });
    previous.disabled = current === 0;
    next.disabled = current === panels.length - 1;
    position.textContent = `${current + 1} / ${panels.length}`;
    if (focus) panels[current].querySelector('h2').focus({preventScroll: true});
  }
  function fromHash() {
    const index = panels.findIndex(panel => `#${panel.id}` === location.hash);
    show(index < 0 ? 0 : index, Boolean(location.hash));
  }
  previous.addEventListener('click', () => { location.hash = panels[current - 1].id; });
  next.addEventListener('click', () => { location.hash = panels[current + 1].id; });
  window.addEventListener('hashchange', fromHash);
  document.querySelector('.pager').hidden = false;
  fromHash();
})();
