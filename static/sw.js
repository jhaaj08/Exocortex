/* Enhanced Service Worker for Seamless Offline Experience */
const CACHE_VERSION = "exocortex-v2.0.0";
const STATIC_CACHE = `${CACHE_VERSION}-static`;
const DYNAMIC_CACHE = `${CACHE_VERSION}-dynamic`;
const OFFLINE_CACHE = `${CACHE_VERSION}-offline`;

console.log("[SW] Enhanced service worker loading...");

// Core assets that should always be cached
const CORE_ASSETS = [
    '/',
    '/offline-home/',
    '/offline-study-enhanced/',
    '/offline-study-session/',
    '/offline-plan/',
    '/focus-blocks/',
    '/static/manifest.json',
    '/static/icons/icon-192.png',
    '/static/icons/icon-512.png'
];

// API endpoints that should be cached when available
const API_ENDPOINTS = [
    '/api/sync-master-sequence/',
    '/api/offline-study-plan/',
    '/api/sync-offline-enhanced/'
];

// Install event - cache core assets
self.addEventListener("install", (event) => {
    console.log("[SW] Installing enhanced service worker...");
    
    event.waitUntil(
        Promise.all([
            // Cache core assets
            caches.open(STATIC_CACHE).then(cache => {
                console.log("[SW] Caching core assets...");
                return cache.addAll(CORE_ASSETS).catch(error => {
                    console.error("[SW] Failed to cache some core assets:", error);
                    // Continue even if some assets fail
                });
            }),
            
            // Initialize offline cache
            caches.open(OFFLINE_CACHE).then(cache => {
                console.log("[SW] Initialized offline cache");
                return cache;
            })
        ]).then(() => {
            console.log("[SW] Installation complete");
            self.skipWaiting(); // Take control immediately
        })
    );
});

// Activate event - clean up old caches and claim clients
self.addEventListener("activate", (event) => {
    console.log("[SW] Activating enhanced service worker...");
    
    event.waitUntil(
        Promise.all([
            // Clean up old caches
            caches.keys().then(cacheNames => {
                return Promise.all(
                    cacheNames.map(cacheName => {
                        if (cacheName !== STATIC_CACHE && 
                            cacheName !== DYNAMIC_CACHE && 
                            cacheName !== OFFLINE_CACHE) {
                            console.log("[SW] Deleting old cache:", cacheName);
                            return caches.delete(cacheName);
                        }
                    })
                );
            }),
            
            // Claim all clients
            self.clients.claim()
        ]).then(() => {
            console.log("[SW] Activation complete - controlling all pages");
        })
    );
});

// Fetch event - comprehensive caching strategy
self.addEventListener("fetch", (event) => {
    const url = new URL(event.request.url);
    const pathname = url.pathname;
    
    // Skip non-GET requests
    if (event.request.method !== 'GET') {
        return;
    }
    
    console.log("[SW] Handling fetch for:", pathname);
    
    // Home page strategy - cache first with offline fallback
    if (pathname === '/' || pathname === '/home/') {
        event.respondWith(homePageStrategy(event.request));
    }
    
    // Offline pages - cache first
    else if (pathname.includes('/offline-')) {
        event.respondWith(cacheFirstStrategy(event.request));
    }
    
    // API endpoints - network first with cache fallback
    else if (pathname.startsWith('/api/')) {
        event.respondWith(networkFirstStrategy(event.request));
    }
    
    // Static assets - cache first
    else if (pathname.startsWith('/static/') || 
             pathname.endsWith('.css') || 
             pathname.endsWith('.js') || 
             pathname.endsWith('.png') || 
             pathname.endsWith('.jpg') || 
             pathname.endsWith('.ico')) {
        event.respondWith(cacheFirstStrategy(event.request));
    }
    
    // Other pages - network first with cache fallback
    else {
        event.respondWith(networkFirstStrategy(event.request));
    }
});

// Home page strategy - seamless offline experience
async function homePageStrategy(request) {
    try {
        // Try network first for fresh data
        const networkResponse = await fetch(request);
        
        if (networkResponse.ok) {
            // Cache the successful response
            const cache = await caches.open(DYNAMIC_CACHE);
            cache.put(request, networkResponse.clone());
            console.log("[SW] Home page cached from network");
            return networkResponse;
        }
        
        throw new Error('Network response not ok');
        
    } catch (error) {
        console.log("[SW] Network failed for home page, trying cache...");
        
        // Try cached version
        const cachedResponse = await caches.match(request);
        if (cachedResponse) {
            console.log("[SW] Serving cached home page");
            return cachedResponse;
        }
        
        // Fallback to offline home page
        console.log("[SW] Serving offline home page");
        const offlineHome = await caches.match('/offline-home/');
        if (offlineHome) {
            return offlineHome;
        }
        
        // Ultimate fallback - basic offline page
        return new Response(`
            <!DOCTYPE html>
            <html>
            <head>
                <title>Exocortex - Offline</title>
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <style>
                    body { font-family: Arial, sans-serif; text-align: center; padding: 50px; }
                    .offline-container { max-width: 500px; margin: 0 auto; }
                    .icon { font-size: 64px; margin-bottom: 20px; }
                    .btn { background: #007bff; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; }
                </style>
            </head>
            <body>
                <div class="offline-container">
                    <div class="icon">📚</div>
                    <h1>Exocortex</h1>
                    <h2>You're Offline</h2>
                    <p>Your study data is cached and ready for offline use.</p>
                    <a href="/offline-study-enhanced/" class="btn">Continue Studying Offline</a>
                </div>
                <script>
                    // Auto-reload when back online
                    window.addEventListener('online', () => {
                        setTimeout(() => window.location.reload(), 1000);
                    });
                </script>
            </body>
            </html>
        `, {
            headers: { 'Content-Type': 'text/html' }
        });
    }
}

// Cache first strategy - for offline pages and static assets
async function cacheFirstStrategy(request) {
    try {
        const cachedResponse = await caches.match(request);
        if (cachedResponse) {
            console.log("[SW] Serving from cache:", request.url);
            return cachedResponse;
        }
        
        console.log("[SW] Cache miss, fetching from network:", request.url);
        const networkResponse = await fetch(request);
        
        if (networkResponse.ok) {
            const cache = await caches.open(DYNAMIC_CACHE);
            cache.put(request, networkResponse.clone());
        }
        
        return networkResponse;
        
    } catch (error) {
        console.error("[SW] Cache first strategy failed:", error);
        
        // For offline pages, return a basic offline message
        if (request.url.includes('/offline-')) {
            return new Response('Offline page not available', { status: 503 });
        }
        
        throw error;
    }
}

// Network first strategy - for API calls and dynamic content
async function networkFirstStrategy(request) {
    try {
        const networkResponse = await fetch(request);
        
        if (networkResponse.ok) {
            // Cache successful API responses
            if (request.url.includes('/api/')) {
                const cache = await caches.open(OFFLINE_CACHE);
                cache.put(request, networkResponse.clone());
            } else {
                const cache = await caches.open(DYNAMIC_CACHE);
                cache.put(request, networkResponse.clone());
            }
        }
        
        return networkResponse;
        
    } catch (error) {
        console.log("[SW] Network failed, trying cache for:", request.url);
        
        const cachedResponse = await caches.match(request);
        if (cachedResponse) {
            console.log("[SW] Serving from cache:", request.url);
            return cachedResponse;
        }
        
        // For API calls, return cached data or error
        if (request.url.includes('/api/')) {
            return new Response(JSON.stringify({
                error: 'Offline - no cached data available',
                offline: true
            }), {
                status: 503,
                headers: { 'Content-Type': 'application/json' }
            });
        }
        
        throw error;
    }
}

// Background sync for offline data
self.addEventListener('sync', (event) => {
    console.log("[SW] Background sync triggered:", event.tag);
    
    if (event.tag === 'sync-offline-progress') {
        event.waitUntil(syncOfflineProgress());
    }
});

// Sync offline progress when connection is restored
async function syncOfflineProgress() {
    try {
        console.log("[SW] Syncing offline progress...");
        
        // This would trigger the sync in the main thread
        const clients = await self.clients.matchAll();
        clients.forEach(client => {
            client.postMessage({
                type: 'SYNC_OFFLINE_PROGRESS'
            });
        });
        
    } catch (error) {
        console.error("[SW] Failed to sync offline progress:", error);
    }
}

// Message handling for communication with main thread
self.addEventListener('message', (event) => {
    console.log("[SW] Received message:", event.data);
    
    if (event.data.type === 'CACHE_STUDY_DATA') {
        // Cache study data for offline use
        cacheStudyData(event.data.data);
    }
});

// Cache study data for offline access
async function cacheStudyData(data) {
    try {
        const cache = await caches.open(OFFLINE_CACHE);
        
        // Store study data as a synthetic response
        const response = new Response(JSON.stringify(data), {
            headers: { 'Content-Type': 'application/json' }
        });
        
        await cache.put('/cached-study-data', response);
        console.log("[SW] Study data cached for offline use");
        
    } catch (error) {
        console.error("[SW] Failed to cache study data:", error);
    }
}

console.log("[SW] Enhanced service worker script loaded successfully");
