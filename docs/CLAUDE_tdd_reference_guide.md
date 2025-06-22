# Test-Driven Development Reference Guide for Claude Code

## Overview

Test-Driven Development (TDD) is a software development technique where you write tests before writing the actual code. This guide provides essential TDD principles, patterns, and code examples specifically for use with Claude Code projects.

## The TDD Cycle: Red-Green-Refactor

### 1. Red - Write a Failing Test
Write a test for the next bit of functionality you want to add. The test should fail initially because the functionality doesn't exist yet.

### 2. Green - Make the Test Pass
Write the minimal amount of code necessary to make the test pass. Don't worry about perfect design at this stage.

### 3. Refactor - Improve the Code
Clean up both the test and production code while keeping all tests passing. Improve design without changing behavior.

## Core TDD Principles

### Start with the Simplest Test
Begin with the most basic test case that exercises the core functionality.

```javascript
// Start simple
test('should return 5 when adding 2 and 3', () => {
  expect(add(2, 3)).toBe(5);
});

// Not this complex test first
test('should handle complex mathematical operations with edge cases', () => {
  // Too complex for first test
});
```

### Write Tests That Drive Design
Tests should express what the code should do, not how it does it.

```python
# Good - expresses intent
def test_user_can_withdraw_money_from_account():
    account = Account(100)
    account.withdraw(50)
    assert account.balance == 50

# Avoid - tests implementation details
def test_withdraw_calls_subtract_method():
    # Testing internal implementation
```

### One Test at a Time
Focus on one failing test before moving to the next. Resist the urge to write multiple tests at once.

## Essential TDD Patterns

### Fake It ('Til You Make It)
Start with the simplest possible implementation, even if it's hardcoded.

```java
// First test
@Test
public void testMultiplyByZero() {
    assertEquals(0, multiply(5, 0));
}

// First implementation (fake it)
public int multiply(int a, int b) {
    return 0;  // Hardcoded to pass the test
}

// Second test forces real implementation
@Test
public void testMultiplyPositiveNumbers() {
    assertEquals(6, multiply(2, 3));
}

// Now implement for real
public int multiply(int a, int b) {
    return a * b;
}
```

### Triangulation
Use multiple test cases to force a general solution.

```python
class TestStringCalculator:
    def test_empty_string_returns_zero(self):
        assert string_add("") == 0
    
    def test_single_number_returns_that_number(self):
        assert string_add("1") == 1
        assert string_add("5") == 5
    
    def test_two_numbers_returns_sum(self):
        assert string_add("1,2") == 3
        assert string_add("3,4") == 7
```

### Obvious Implementation
When the implementation is obvious, write it directly.

```javascript
// When it's obvious, just implement it
function isEmpty(array) {
  return array.length === 0;
}

test('empty array should return true', () => {
  expect(isEmpty([])).toBe(true);
});

test('non-empty array should return false', () => {
  expect(isEmpty([1, 2, 3])).toBe(false);
});
```

## Test Organization Patterns

### Arrange-Act-Assert (AAA)
Structure tests clearly with three distinct sections.

```csharp
[Test]
public void ShouldCalculateOrderTotal()
{
    // Arrange
    var order = new Order();
    order.AddItem(new Item("Widget", 10.00m));
    order.AddItem(new Item("Gadget", 15.00m));
    
    // Act
    var total = order.CalculateTotal();
    
    // Assert
    Assert.AreEqual(25.00m, total);
}
```

### Given-When-Then (BDD Style)
Express tests in business language.

```python
def test_account_withdrawal():
    # Given an account with a balance of 100
    account = Account(100)
    
    # When I withdraw 30
    account.withdraw(30)
    
    # Then the balance should be 70
    assert account.balance == 70
```

## Testing Different Scenarios

### Testing Exceptions
Test error conditions explicitly.

```java
@Test(expected = InsufficientFundsException.class)
public void shouldThrowExceptionWhenWithdrawingMoreThanBalance() {
    Account account = new Account(50);
    account.withdraw(100);
}

// Or with modern JUnit
@Test
public void shouldThrowExceptionWhenWithdrawingMoreThanBalance() {
    Account account = new Account(50);
    assertThrows(InsufficientFundsException.class, () -> {
        account.withdraw(100);
    });
}
```

### Testing Edge Cases
Include boundary conditions and edge cases.

```javascript
describe('Password Validator', () => {
  test('should reject passwords shorter than 8 characters', () => {
    expect(isValidPassword('1234567')).toBe(false);
  });
  
  test('should accept passwords exactly 8 characters', () => {
    expect(isValidPassword('12345678')).toBe(true);
  });
  
  test('should handle empty password', () => {
    expect(isValidPassword('')).toBe(false);
  });
  
  test('should handle null password', () => {
    expect(isValidPassword(null)).toBe(false);
  });
});
```

## Refactoring Patterns

### Extract Method
When tests pass, extract complex logic into well-named methods.

```python
# Before refactoring
def calculate_shipping(weight, distance, is_express):
    if is_express:
        base_cost = weight * 2.5
        if distance > 100:
            base_cost += distance * 0.1
        else:
            base_cost += 10
        return base_cost * 1.5
    else:
        base_cost = weight * 1.2
        if distance > 100:
            base_cost += distance * 0.05
        else:
            base_cost += 5
        return base_cost

# After refactoring
def calculate_shipping(weight, distance, is_express):
    if is_express:
        return calculate_express_shipping(weight, distance)
    else:
        return calculate_standard_shipping(weight, distance)

def calculate_express_shipping(weight, distance):
    base_cost = weight * 2.5
    distance_cost = calculate_distance_cost(distance, 0.1, 10)
    return (base_cost + distance_cost) * 1.5

def calculate_standard_shipping(weight, distance):
    base_cost = weight * 1.2
    distance_cost = calculate_distance_cost(distance, 0.05, 5)
    return base_cost + distance_cost

def calculate_distance_cost(distance, rate, base_fee):
    return distance * rate if distance > 100 else base_fee
```

### Remove Duplication
Eliminate duplicate code while keeping tests green.

```javascript
// Before - duplication in tests
test('should format USD currency', () => {
  const formatter = new CurrencyFormatter();
  expect(formatter.format(100, 'USD')).toBe('$100.00');
});

test('should format EUR currency', () => {
  const formatter = new CurrencyFormatter();
  expect(formatter.format(100, 'EUR')).toBe('€100.00');
});

// After - extract setup
describe('CurrencyFormatter', () => {
  let formatter;
  
  beforeEach(() => {
    formatter = new CurrencyFormatter();
  });
  
  test('should format USD currency', () => {
    expect(formatter.format(100, 'USD')).toBe('$100.00');
  });
  
  test('should format EUR currency', () => {
    expect(formatter.format(100, 'EUR')).toBe('€100.00');
  });
});
```

## TDD Anti-Patterns to Avoid

### Writing Too Many Tests at Once
**Problem**: Writing multiple failing tests before implementing any code.
**Solution**: Focus on one red test at a time.

### Testing Implementation Details
**Problem**: Tests that break when you refactor internal code structure.
**Solution**: Test behavior and outcomes, not internal methods.

### Skipping the Refactor Step
**Problem**: Accumulating technical debt by not cleaning up code.
**Solution**: Always refactor when tests are green.

### Writing Tests After Code
**Problem**: Tests that don't drive design and may miss edge cases.
**Solution**: Always write the test first.

## Sample TDD Session: Building a Shopping Cart

```python
# Test 1: Empty cart
def test_empty_cart_has_zero_total():
    cart = ShoppingCart()
    assert cart.total() == 0

# Implementation 1
class ShoppingCart:
    def total(self):
        return 0

# Test 2: Single item
def test_single_item_cart():
    cart = ShoppingCart()
    cart.add_item("apple", 1.50)
    assert cart.total() == 1.50

# Implementation 2
class ShoppingCart:
    def __init__(self):
        self.items = []
    
    def add_item(self, name, price):
        self.items.append({"name": name, "price": price})
    
    def total(self):
        return sum(item["price"] for item in self.items)

# Test 3: Multiple items
def test_multiple_items_cart():
    cart = ShoppingCart()
    cart.add_item("apple", 1.50)
    cart.add_item("banana", 0.75)
    assert cart.total() == 2.25

# Implementation 3: Already works!

# Test 4: Remove items
def test_remove_item_from_cart():
    cart = ShoppingCart()
    cart.add_item("apple", 1.50)
    cart.add_item("banana", 0.75)
    cart.remove_item("apple")
    assert cart.total() == 0.75

# Implementation 4
class ShoppingCart:
    def __init__(self):
        self.items = []
    
    def add_item(self, name, price):
        self.items.append({"name": name, "price": price})
    
    def remove_item(self, name):
        self.items = [item for item in self.items if item["name"] != name]
    
    def total(self):
        return sum(item["price"] for item in self.items)
```

## Key Reminders for Claude Code

1. **Always start with a failing test** - Red first, then Green
2. **Write the minimal code** to make the test pass
3. **Refactor ruthlessly** when tests are green
4. **One test at a time** - resist writing multiple failing tests
5. **Test behavior, not implementation** - focus on what, not how
6. **Use descriptive test names** that explain the expected behavior
7. **Keep tests simple and focused** - one assertion per test when possible
8. **Don't skip the refactor step** - clean code is as important as working code

## Resources

- **Book**: "Test Driven Development: By Example" by Kent Beck
- **Online**: Martin Fowler's article on TDD
- **Practice**: Code katas (String Calculator, Bowling Game, Roman Numerals)
- **Framework docs**: Jest (JavaScript), pytest (Python), JUnit (Java), etc.

Remember: TDD is about confidence, design, and maintainability. The tests are a safety net that allows you to refactor fearlessly and catch regressions early.