# Web Development Fundamentals

## HTML (HyperText Markup Language)

HTML is the backbone of web pages. It provides structure and content.

### Basic HTML Structure

```html
<!DOCTYPE html>
<html>
<head>
    <title>Page Title</title>
</head>
<body>
    <h1>Main Heading</h1>
    <p>This is a paragraph.</p>
</body>
</html>
```

### Common HTML Elements

- `<h1>` to `<h6>`: Headings
- `<p>`: Paragraphs
- `<div>`: Generic container
- `<span>`: Inline container
- `<a>`: Links
- `<img>`: Images
- `<ul>` and `<li>`: Lists

## CSS (Cascading Style Sheets)

CSS is used to style and layout web pages.

### CSS Selectors

- **Element selector**: `p { color: blue; }`
- **Class selector**: `.my-class { font-size: 16px; }`
- **ID selector**: `#my-id { background: red; }`

### CSS Box Model

Every HTML element is a rectangular box consisting of:

1. **Content**: The actual content
2. **Padding**: Space inside the element
3. **Border**: Edge of the element
4. **Margin**: Space outside the element

## JavaScript

JavaScript adds interactivity to web pages.

### Variables

```javascript
let name = "John";
const age = 30;
var city = "New York";
```

### Functions

```javascript
function calculateArea(width, height) {
    return width * height;
}

const area = calculateArea(10, 5);
```

### DOM Manipulation

```javascript
// Select an element
const element = document.getElementById("myElement");

// Change content
element.textContent = "New content";

// Add event listener
element.addEventListener("click", function() {
    alert("Element clicked!");
});
```

## Modern Web Development

### Frameworks and Libraries

- **Frontend**: React, Vue.js, Angular
- **Backend**: Node.js, Django, Flask
- **CSS**: Bootstrap, Tailwind CSS

### Development Tools

- **Version Control**: Git and GitHub
- **Package Managers**: npm, yarn
- **Build Tools**: Webpack, Vite
- **Code Editors**: VS Code, WebStorm 