const CACHE_NAME = "alarm-wake-challenge-v6";
const CACHE_PATHS = [
  "./",
  "./index.html",
  "./manifest.webmanifest",
  "./src/app.js?v=20260702-3",
  "./src/styles.css?v=20260702-3",
  "./assets/icon-192.png",
  "./assets/icon-512.png",
  "./assets/maskable-512.png",
  "./models/pose_landmarker_lite.task",
  "./vendor/mediapipe/vision_bundle.mjs",
  "./vendor/mediapipe/wasm/vision_wasm_internal.js",
  "./vendor/mediapipe/wasm/vision_wasm_internal.wasm",
  "./vendor/mediapipe/wasm/vision_wasm_module_internal.js",
  "./vendor/mediapipe/wasm/vision_wasm_module_internal.wasm",
  "./vendor/mediapipe/wasm/vision_wasm_nosimd_internal.js",
  "./vendor/mediapipe/wasm/vision_wasm_nosimd_internal.wasm"
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches
      .open(CACHE_NAME)
      .then((cache) =>
        cache.addAll(
          CACHE_PATHS.map((path) => new URL(path, self.registration.scope).href)
        )
      )
      .then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(
          keys
            .filter((key) => key !== CACHE_NAME)
            .map((key) => caches.delete(key))
        )
      )
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  if (event.request.method !== "GET") {
    return;
  }

  if (event.request.mode === "navigate") {
    event.respondWith(
      fetch(event.request)
        .then((response) => {
          const copy = response.clone();
          caches.open(CACHE_NAME).then((cache) => {
            cache.put(event.request, copy.clone());
            cache.put(new URL("./index.html", self.registration.scope).href, copy);
          });
          return response;
        })
        .catch(() =>
          caches.match(event.request).then(
            (cached) =>
              cached ||
              caches.match(new URL("./index.html", self.registration.scope).href)
          )
        )
    );
    return;
  }

  event.respondWith(
    caches.match(event.request).then((cached) => {
      if (cached) {
        return cached;
      }

      return fetch(event.request)
        .then((response) => {
          if (!response || response.status !== 200 || response.type === "opaque") {
            return response;
          }

          const requestUrl = new URL(event.request.url);
          const scopeUrl = new URL(self.registration.scope);
          if (requestUrl.origin === scopeUrl.origin) {
            const copy = response.clone();
            caches.open(CACHE_NAME).then((cache) => cache.put(event.request, copy));
          }
          return response;
        })
        .catch(() => Response.error());
    })
  );
});
