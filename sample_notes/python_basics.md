# Python Basics

## Variables and Data Types

Python supports several built-in data types:

- **Strings**: Text data enclosed in quotes
- **Integers**: Whole numbers like 42
- **Floats**: Decimal numbers like 3.14
- **Booleans**: True or False values
- **Lists**: Ordered collections of items
- **Dictionaries**: Key-value pairs

### Variable Assignment

Variables in Python are created when you assign a value to them:

```python
name = "Alice"
age = 25
height = 5.6
is_student = True
```

## Control Structures

### If Statements

Use if statements to make decisions in your code:

```python
if age >= 18:
    print("You are an adult")
else:
    print("You are a minor")
```

### Loops

Python has two main types of loops:

1. **For loops**: Iterate over sequences
2. **While loops**: Continue while a condition is true

```python
# For loop
for i in range(5):
    print(i)

# While loop
count = 0
while count < 5:
    print(count)
    count += 1
```

## Functions

Functions are reusable blocks of code:

```python
def greet(name):
    return f"Hello, {name}!"

message = greet("World")
print(message)
```

## Key Concepts

- **Indentation**: Python uses indentation to define code blocks
- **Dynamic typing**: Variables don't need explicit type declarations
- **Object-oriented**: Everything in Python is an object
- **Interpreted**: Python code is executed line by line 