const VERSION = 'repo-s2-tmi-v2';
const CORE = [
  './',
  './index.html',
  './logo.png',
  './zaa.svg',
  './manifest.json',
  './icons/icon-192.png',
  './icons/icon-512.png',
  './offline.html'
];

self.addEventListener('install', e => {
  e.waitUntil(
    caches.open(VERSION).then(c =>
      // tolerate a missing file so install never fails
      Promise.all(CORE.map(u => c.add(u).catch(() => {})))
    ).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys()
      .then(keys => Promise.all(keys.filter(k => k !== VERSION).map(k => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', e => {
  const req = e.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);

  // Google Fonts: cache-first
  if (url.hostname === 'fonts.googleapis.com' || url.hostname === 'fonts.gstatic.com') {
    e.respondWith(
      caches.match(req).then(hit => hit || fetch(req).then(res => {
        const copy = res.clone();
        caches.open(VERSION).then(c => c.put(req, copy));
        return res;
      }))
    );
    return;
  }

  // Other origins (e.g. thesis catalog): leave to the network
  if (url.origin !== location.origin) return;

  // Pages and JSON data (e.g. bahan/manifest.json file list): network-first
  if (req.mode === 'navigate' || url.pathname.endsWith('.json')) {
    e.respondWith(
      fetch(req).then(res => {
        const copy = res.clone();
        caches.open(VERSION).then(c => c.put(req, copy));
        return res;
      }).catch(() =>
        caches.match(req).then(hit => hit || (req.mode === 'navigate' ? caches.match('./offline.html') : Response.error()))
      )
    );
    return;
  }

  // Static assets: stale-while-revalidate
  e.respondWith(
    caches.match(req).then(hit => {
      const net = fetch(req).then(res => {
        if (res && res.ok) {
          const copy = res.clone();
          caches.open(VERSION).then(c => c.put(req, copy));
        }
        return res;
      }).catch(() => hit);
      return hit || net;
    })
  );
});
