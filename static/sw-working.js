/* Exocortex PWA Service Worker - Working Version */
const CACHE_VERSION = "exocortex-v1.3.0";
const STATIC_CACHE = `${CACHE_VERSION}-static`;
const DYNAMIC_CACHE = `${CACHE_VERSION}-dynamic`;

// Core assets to cache - only essential ones that we know work
const CORE_ASSETS = [
  "/",
  "/offline-home/",
  "/offline-plan/",
  "/static/manifest.json"
];

// Optional assets to try caching (won't fail installation if they fail)
const OPTIONAL_ASSETS = [
  "/static/icons/icon-192.png",
  "/static/icons/icon-512.png"
];

// Install event - cache core assets with robust error handling
self.addEventListener("install", (event) => {
  console.log("[SW] Installing service worker v1.3.0...");
  
  event.waitUntil(
    caches.open(STATIC_CACHE).then(async (cache) => {
      console.log("[SW] Caching core assets...");
      
      try {
        // Cache essential assets - these must succeed
        await cache.addAll(CORE_ASSETS);
        console.log("[SW] ✅ Core assets cached successfully");
        
        // Try to cache optional assets - failures won't break installation
        for (const asset of OPTIONAL_ASSETS) {
          try {
            await cache.add(asset);
            console.log(`[SW] ✅ Optional asset cached: ${asset}`);
          } catch (error) {
            console.log(`[SW] ⚠️ Optional asset failed (continuing): ${asset}`, error.message);
          }
        }
        
        console.log("[SW] Installation completed successfully");
        
      } catch (error) {
        console.error("[SW] ❌ Core assets caching failed:", error);
        throw error; // This will cause installation to fail
      }
    })
  );
  
  self.skipWaiting(); // Force activation
});

// Activate event - clean up old caches
self.addEventListener("activate", (event) => {
  console.log("[SW] Activating service worker...");
  
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      const deletePromises = cacheNames
        .filter((cacheName) => !cacheName.startsWith(CACHE_VERSION))
        .map((cacheName) => {
          console.log("[SW] Deleting old cache:", cacheName);
          return caches.delete(cacheName);
        });
      
      return Promise.all(deletePromises);
    }).then(() => {
      console.log("[SW] ✅ Service worker activated successfully");
      return self.clients.claim(); // Take control immediately
    })
  );
});

// Fetch event - simplified handling
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

  // Simple caching strategy
  event.respondWith(handleRequest(request));
});

// Unified request handler
async function handleRequest(request) {
  try {
    // Try cache first for offline pages
    if (request.url.includes("/offline-") || request.url.includes("/static/")) {
      const cachedResponse = await caches.match(request);
      if (cachedResponse) {
        return cachedResponse;
      }
    }
    
    // Try network
    const networkResponse = await fetch(request);
    
    // Cache successful responses
    if (networkResponse.status === 200) {
      const cache = await caches.open(DYNAMIC_CACHE);
      cache.put(request, networkResponse.clone());
    }
    
    return networkResponse;
    
  } catch (error) {
    console.log("[SW] Network failed, trying cache:", error.message);
    
    // Try cache as fallback
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
      return cachedResponse;
    }
    
    // Special fallbacks for key pages
    const url = new URL(request.url);
    
    if (url.pathname === "/" || url.pathname.includes("/focus-blocks/")) {
      const offlineHome = await caches.match("/offline-home/");
      if (offlineHome) {
        return offlineHome;
      }
    }
    
    // Ultimate fallback
    return new Response("Offline - Content not available", { 
      status: 503,
      headers: { 'Content-Type': 'text/plain' }
    });
  }
}

console.log("[SW] ✅ Service worker script loaded");
