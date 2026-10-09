const VERSION = new URL(self.location.href).searchParams.get("v") || "dev";
const CACHE = `antifive-pyodide-${VERSION}`;
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

async function networkFirst(request) {
  const cache = await caches.open(CACHE);
  try {
    const response = await fetch(request);
    if (response.ok) cache.put(request, response.clone());
    return response;
  } catch (error) {
    const hit = await cache.match(request);
    if (hit) return hit;
    throw error;
  }
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
    event.respondWith(networkFirst(event.request));
    return;
  }
  event.respondWith(cacheFirst(event.request));
});
