/* Simple Service Worker for Testing */
console.log('Service Worker loaded');

const CACHE_NAME = 'test-cache-v1';

self.addEventListener('install', (event) => {
  console.log('[SW] Install event');
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  console.log('[SW] Activate event');
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  console.log('[SW] Fetch event for:', event.request.url);
  // Just pass through for now
  event.respondWith(fetch(event.request));
});
