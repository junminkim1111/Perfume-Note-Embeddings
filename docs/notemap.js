// 3D note map in the hero: notes.json (made by src/make_figures.py note_map_3d) -> Plotly scatter3d
(() => {
  const el = document.getElementById('notemap');
  const card = document.getElementById('note-card');
  const ko = document.documentElement.lang === 'ko';
  const T = ko
    ? { fam: ['시트러스·프루티', '플로럴', '그린·아로마틱', '우디·앰버', '구르망'], other: '기타', nearest: '가장 가까운 노트', none: '향 설명 없음 (UNKNOWN)' }
    : { fam: null, other: 'Other', nearest: 'Nearest notes', none: 'No description (UNKNOWN)' };
  const wide = el.clientWidth > 860;  // wide: cloud a bit right of centre, legend top right; narrow: cloud in the lower part, legend at the bottom
  const dom = wide ? { x: [0.2, 1], y: [0, 1] } : { x: [0, 1], y: [0, 0.6] };
  const aspect = el.clientHeight * (dom.y[1] - dom.y[0]) / (el.clientWidth * (dom.x[1] - dom.x[0]));
  const OTHER = '#6f7d77', R = 1.25 * Math.max(1, 0.85 * aspect), R0 = 3;  // tall scene (phone): camera further back

  fetch('notes.json').then(r => r.json()).then(d => {
    const N = d.notes;
    const famName = k => k < 0 ? T.other : (T.fam ? T.fam[k] : d.families[k]);
    const famColor = k => k < 0 ? OTHER : d.colors[k];
    const col = (ids, c) => ids.map(i => N[i][c]);

    const traces = [-1, ...d.families.keys()].map(k => {
      const ids = N.flatMap((n, i) => n[4] === k ? [i] : []);
      return {
        type: 'scatter3d', mode: 'markers', name: famName(k),
        x: col(ids, 1), y: col(ids, 2), z: col(ids, 3), text: col(ids, 0), customdata: ids,
        hovertemplate: '%{text}<extra></extra>',
        marker: { size: 3, color: famColor(k) },
      };
    });
    const LINES = traces.length, HL = traces.length + 1;  // selected note + its nearest notes
    traces.push({ type: 'scatter3d', mode: 'lines', x: [], y: [], z: [], hoverinfo: 'skip', showlegend: false,
                  line: { color: 'rgba(238,243,239,.55)', width: 2 } });
    traces.push({ type: 'scatter3d', mode: 'markers+text', x: [], y: [], z: [], text: [], customdata: [], showlegend: false,
                  textposition: 'top center', textfont: { color: '#eef3ef', size: 12 }, hovertemplate: '%{text}<extra></extra>',
                  marker: { size: [], color: [] } });

    const axis = { visible: false, range: [-1.05, 1.05] };
    const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
    let angle = Math.PI / 4;
    const eye = r => ({ x: r * Math.cos(angle), y: r * Math.sin(angle), z: 0.4 * r / R });
    Plotly.newPlot(el, traces, {
      paper_bgcolor: 'rgba(0,0,0,0)', margin: { l: 0, r: 0, t: 0, b: 0 },
      legend: { orientation: 'h', itemsizing: 'constant', font: { color: '#a7b8b0', size: 12 },
                ...(wide ? { x: 1, xanchor: 'right', y: 0.9 } : { x: 0.04, y: 0.02, yanchor: 'bottom' }) },
      hoverlabel: { bgcolor: '#15241f', bordercolor: '#8fd1b5', font: { color: '#eef3ef' } },
      scene: { xaxis: axis, yaxis: axis, zaxis: axis, aspectmode: 'cube', dragmode: 'turntable',
               domain: dom,
               camera: { eye: eye(reduce ? R : R0) } },
    }, { displayModeBar: false, scrollZoom: false, responsive: true });

    // Intro: points spread out as the camera moves in, then a slow turn until the user touches the map
    let rotating = !reduce, visible = true, hovering = false;
    const t0 = performance.now();
    new IntersectionObserver(([e]) => { visible = e.isIntersecting; }).observe(el);
    ['mousedown', 'touchstart'].forEach(ev => el.addEventListener(ev, () => { rotating = false; }, { passive: true }));
    el.addEventListener('mouseenter', () => { hovering = true; });  // a moving camera makes points hard to hover
    el.addEventListener('mouseleave', () => { hovering = false; });
    (function frame(now) {
      if (!rotating) return;
      if (visible && !(hovering && now - t0 > 1600)) {  // always finish the intro
        const t = Math.min((now - t0) / 1600, 1);
        angle += 0.002;
        Plotly.relayout(el, { 'scene.camera.eye': eye(R0 - (R0 - R) * (1 - (1 - t) ** 3)) });
      }
      requestAnimationFrame(frame);
    })(t0);

    function select(i) {
      rotating = false;
      const nn = d.nn[i], ids = [i, ...nn.map(p => p[0])];
      const seg = c => nn.flatMap(([j]) => [N[i][c], N[j][c], null]);
      Plotly.restyle(el, { x: [seg(1)], y: [seg(2)], z: [seg(3)] }, [LINES]);
      Plotly.restyle(el, {
        x: [col(ids, 1)], y: [col(ids, 2)], z: [col(ids, 3)], text: [col(ids, 0)], customdata: [ids],
        'marker.size': [ids.map((_, k) => k ? 6 : 10)], 'marker.color': [ids.map((_, k) => k ? '#8fd1b5' : '#ffffff')],
      }, [HL]);

      const k = N[i][4];
      card.querySelector('.nc-fam').innerHTML = `<i style="background:${famColor(k)}"></i>`;
      card.querySelector('.nc-fam').append(famName(k));
      card.querySelector('.nc-name').textContent = N[i][0];
      card.querySelector('.nc-desc').textContent = d.desc[i] || T.none;
      card.querySelector('.nc-label').textContent = T.nearest;
      card.querySelector('.nc-nn').replaceChildren(...nn.map(([j, s]) => {
        const li = document.createElement('li'), b = document.createElement('button'), sim = document.createElement('span');
        b.textContent = N[j][0];
        sim.textContent = s.toFixed(2);
        b.append(sim);
        b.onclick = () => select(j);
        li.append(b);
        return li;
      }));
      card.hidden = false;
    }

    el.on('plotly_click', e => {
      const p = e.points[0];
      if (p && p.customdata != null) select(p.customdata);
    });
    card.querySelector('.nc-close').onclick = () => {
      card.hidden = true;
      Plotly.restyle(el, { x: [[]], y: [[]], z: [[]], text: [[]], customdata: [[]] }, [LINES, HL]);
    };
  });
})();
