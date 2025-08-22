/* Root Scope Service Worker - Controls All Pages */
const CACHE_VERSION = "exocortex-root-v1";

console.log("[SW] Root scope service worker loading...");

// Install event - activate immediately
self.addEventListener("install", (event) => {
  console.log("[SW] Installing root scope service worker...");
  self.skipWaiting(); // Take control immediately
});

// Activate event - claim all clients
self.addEventListener("activate", (event) => {
  console.log("[SW] Activating root scope service worker...");
  event.waitUntil(self.clients.claim()); // Control existing pages
  console.log("[SW] Root scope service worker activated and claimed all clients");
});

// Fetch event - handle all requests
self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  console.log("[SW] Handling fetch for:", url.pathname);
  
  // For offline pages, try cache first
  if (url.pathname.includes("/offline-")) {
    event.respondWith(
      caches.match(event.request).then(cachedResponse => {
        if (cachedResponse) {
          console.log("[SW] Serving from cache:", url.pathname);
          return cachedResponse;
        }
        console.log("[SW] Cache miss, fetching from network:", url.pathname);
        return fetch(event.request);
      })
    );
  }
  // For other requests, just pass through for now
});

console.log("[SW] Root scope service worker script loaded successfully");
