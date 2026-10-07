const CACHE = "antifive-pyodide-314.0.7";
const MIRROR_HOSTS = ["cdn.npmmirror.com", "cdn.jsdelivr.net", "fastly.jsdelivr.net"];

self.addEventListener("install", () => {
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) => Promise.all(keys.filter((key) => key !== CACHE).map((key) => caches.delete(key))))
      .then(() => self.clients.claim()),
  );
});

function cacheFirst(request) {
  return caches.open(CACHE).then(async (cache) => {
    const hit = await cache.match(request);
    if (hit) return hit;
    const response = await fetch(request);
    if (response.ok) cache.put(request, response.clone());
    return response;
  });
}

self.addEventListener("fetch", (event) => {
  if (event.request.method !== "GET") return;
  const url = new URL(event.request.url);
  const sameOrigin = url.origin === self.location.origin;
  const mirror = MIRROR_HOSTS.includes(url.hostname);
  if (!sameOrigin && !mirror) return;
  if (sameOrigin && !url.pathname.endsWith("/antifive.whl") && !url.pathname.includes("/pyodide/")) {
    return;
  }
  if (url.pathname.endsWith("/antifive.whl")) {
    event.respondWith(
      caches.open(CACHE).then(async (cache) => {
        const network = fetch(event.request).then((response) => {
          if (response.ok) cache.put(event.request, response.clone());
          return response;
        });
        const hit = await cache.match(event.request);
        return hit ?? network;
      }),
    );
    return;
  }
  event.respondWith(cacheFirst(event.request));
});
