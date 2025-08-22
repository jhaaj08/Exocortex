/* Exocortex PWA Service Worker */
const CACHE_VERSION = "exocortex-v1.1.0";
const STATIC_CACHE = `${CACHE_VERSION}-static`;
const DYNAMIC_CACHE = `${CACHE_VERSION}-dynamic`;

// Core assets to cache immediately
const CORE_ASSETS = [
  "/",
  "/offline-home/",
  "/offline-plan/",
  "/static/manifest.json",
  "/static/icons/icon-192.png",
  "/static/icons/icon-512.png",
  // Add your critical CSS/JS files here when you have them
];

// Install event - cache core assets
self.addEventListener("install", (event) => {
  console.log("[SW] Installing service worker...");
  event.waitUntil(
    caches.open(STATIC_CACHE).then((cache) => {
      console.log("[SW] Caching core assets");
      return cache.addAll(CORE_ASSETS);
    })
  );
  self.skipWaiting(); // Force activation
});

// Activate event - clean up old caches
self.addEventListener("activate", (event) => {
  console.log("[SW] Activating service worker...");
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames
          .filter((cacheName) => !cacheName.startsWith(CACHE_VERSION))
          .map((cacheName) => {
            console.log("[SW] Deleting old cache:", cacheName);
            return caches.delete(cacheName);
          })
      );
    })
  );
  self.clients.claim(); // Take control immediately
});

// Fetch event - serve from cache with network fallback
self.addEventListener("fetch", (event) => {
  const { request } = event;
  
  // Skip non-GET requests
  if (request.method !== "GET") {
    return;
  }

  // Skip external requests
  if (!request.url.startsWith(self.location.origin)) {
    return;
  }

  // Handle different types of requests
  if (request.url.includes("/static/")) {
    // Static files - cache first strategy
    event.respondWith(cacheFirstStrategy(request));
  } else if (request.url.includes("/api/") || request.url.includes("/admin/")) {
    // API/Admin - network first strategy
    event.respondWith(networkFirstStrategy(request));
  } else if (request.url.includes("/offline-plan/") || request.url.includes("/offline-home/") || request.url.includes("/export/study-pack/")) {
    // Offline study features - cache first strategy
    event.respondWith(cacheFirstStrategy(request));
  } else if (request.url.endsWith("/") || request.url.includes("/focus-blocks/")) {
    // Home page and study plan - special offline handling
    event.respondWith(homePageStrategy(request));
  } else {
    // Other HTML pages - stale while revalidate strategy
    event.respondWith(staleWhileRevalidateStrategy(request));
  }
});

// Cache first strategy for static assets
async function cacheFirstStrategy(request) {
  try {
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
      return cachedResponse;
    }
    
    const networkResponse = await fetch(request);
    if (networkResponse.status === 200) {
      const cache = await caches.open(STATIC_CACHE);
      cache.put(request, networkResponse.clone());
    }
    return networkResponse;
  } catch (error) {
    console.log("[SW] Cache first failed:", error);
    return new Response("Offline", { status: 503 });
  }
}

// Network first strategy for dynamic content
async function networkFirstStrategy(request) {
  try {
    const networkResponse = await fetch(request);
    if (networkResponse.status === 200) {
      const cache = await caches.open(DYNAMIC_CACHE);
      cache.put(request, networkResponse.clone());
    }
    return networkResponse;
  } catch (error) {
    console.log("[SW] Network first failed, trying cache:", error);
    const cachedResponse = await caches.match(request);
    return cachedResponse || new Response("Offline", { status: 503 });
  }
}

// Stale while revalidate strategy for HTML pages
async function staleWhileRevalidateStrategy(request) {
  try {
    const cachedResponse = await caches.match(request);
    
    const networkResponsePromise = fetch(request).then(async (networkResponse) => {
      if (networkResponse.status === 200) {
        const cache = await caches.open(DYNAMIC_CACHE);
        cache.put(request, networkResponse.clone());
      }
      return networkResponse;
    });

    // Return cached version immediately, update in background
    if (cachedResponse) {
      networkResponsePromise.catch(() => {}); // Ignore network errors
      return cachedResponse;
    }

    // If no cache, wait for network
    return await networkResponsePromise;
  } catch (error) {
    console.log("[SW] Stale while revalidate failed:", error);
    
    // Try to return cached fallback
    const fallback = await caches.match("/");
    if (fallback) {
      return fallback;
    }
    
    // Ultimate fallback - redirect to offline plan
    return Response.redirect("/offline-plan/", 302);
  }
}

// Home page strategy - serves cached home page or offline home page
async function homePageStrategy(request) {
  try {
    // Try network first for fresh content
    const networkResponse = await fetch(request);
    if (networkResponse.status === 200) {
      const cache = await caches.open(DYNAMIC_CACHE);
      cache.put(request, networkResponse.clone());
      return networkResponse;
    }
  } catch (error) {
    console.log("[SW] Network failed for home page, serving offline version:", error);
  }
  
  // Network failed, serve appropriate offline page
  const url = new URL(request.url);
  
  if (url.pathname.includes("/focus-blocks/")) {
    // Redirect study plan to offline version
    const offlinePlan = await caches.match("/offline-plan/");
    if (offlinePlan) {
      return offlinePlan;
    }
  }
  
  if (url.pathname === "/" || url.pathname === "/home/") {
    // Serve offline home page for root requests
    const offlineHome = await caches.match("/offline-home/");
    if (offlineHome) {
      return offlineHome;
    }
    
    // Fallback to cached home page
    const cachedHome = await caches.match("/");
    if (cachedHome) {
      return cachedHome;
    }
  }
  
  // Last resort: redirect to offline plan
  return Response.redirect("/offline-plan/", 302);
}

// Background sync for future features
self.addEventListener("sync", (event) => {
  console.log("[SW] Background sync:", event.tag);
  // Add background sync logic here if needed
});

// Push notifications for future features
self.addEventListener("push", (event) => {
  console.log("[SW] Push notification received");
  // Add push notification logic here if needed
}); 