/* Minimal Service Worker - No Installation Caching */
const CACHE_VERSION = "exocortex-minimal-v1";

console.log("[SW] Minimal service worker loading...");

// Install event - skip caching, just activate immediately
self.addEventListener("install", (event) => {
  console.log("[SW] Installing minimal service worker...");
  console.log("[SW] Skipping cache setup for now");
  self.skipWaiting();
});

// Activate event - just claim clients
self.addEventListener("activate", (event) => {
  console.log("[SW] Activating minimal service worker...");
  event.waitUntil(self.clients.claim());
  console.log("[SW] Minimal service worker activated successfully");
});

// Fetch event - just pass through to network for now
self.addEventListener("fetch", (event) => {
  // Just let requests go through normally
  console.log("[SW] Handling fetch for:", event.request.url);
});

console.log("[SW] Minimal service worker script loaded successfully");
