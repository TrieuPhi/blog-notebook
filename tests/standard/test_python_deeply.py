"""
Unit tests for Python Internals: Descriptors, Metaclasses, and Garbage Collection.
Covers the implementation in learning-python-deeply.md.
"""
from __future__ import annotations

import gc
import unittest
from typing import Any, ClassVar

# ==============================================================================
# TEST IMPLEMENTATIONS CO-LOCATED FOR ISOLATED, DETERMINISTIC EXECUTION
# ==============================================================================

class PositiveInteger:
    """
    Data Descriptor enforcing positive integer constraints.
    Supports PEP 487 __set_name__.
    """

    def __init__(self, default: int = 1) -> None:
        self.default = default
        self.storage_name: str = ""

    def __set_name__(self, owner: type[Any], name: str) -> None:
        self.storage_name = f"_val_{name}"

    def __get__(self, instance: Any, owner: type[Any] | None = None) -> Any:
        if instance is None:
            return self
        return getattr(instance, self.storage_name, self.default)

    def __set__(self, instance: Any, value: Any) -> None:
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"Attribute {self.storage_name} must be an integer, got {type(value).__name__}")
        if value <= 0:
            raise ValueError(f"Attribute {self.storage_name} must be > 0, got {value}")
        setattr(instance, self.storage_name, value)

    def __delete__(self, instance: Any) -> None:
        if hasattr(instance, self.storage_name):
            delattr(instance, self.storage_name)


class NonDataMethodDescriptor:
    """
    Non-data Descriptor implementing only __get__.
    Can be shadowed by instance dictionary assignments.
    """

    def __init__(self, func) -> None:
        self.func = func

    def __get__(self, instance: Any, owner: type[Any] | None = None) -> Any:
        if instance is None:
            return self.func
        return lambda *args, **kwargs: self.func(instance, *args, **kwargs)


class PluginRegistryMeta(type):
    """
    Metaclass that enforces the `execute` method interface and
    automatically registers subclasses into a central dictionary.
    """
    registry: ClassVar[dict[str, type[Any]]] = {}

    def __new__(metacls, name: str, bases: tuple, namespace: dict, **kwargs):
        cls_instance = super().__new__(metacls, name, bases, namespace)
        if name != "BaseTestPlugin":
            if "execute" not in namespace or not callable(namespace["execute"]):
                raise TypeError(f"Plugin class '{name}' must implement a callable 'execute()' method")
            plugin_id = kwargs.get("plugin_id", name.lower())
            cls_instance._plugin_id = plugin_id
            metacls.registry[plugin_id] = cls_instance
        return cls_instance


class BaseTestPlugin(metaclass=PluginRegistryMeta):
    """Abstract base plugin for testing."""


class CyclicNode:
    """Helper class to demonstrate cyclic reference garbage collection."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.peer: CyclicNode | None = None

    def connect(self, other: CyclicNode) -> None:
        self.peer = other


# ==============================================================================
# UNIT TEST CASES
# ==============================================================================

class TestDescriptorProtocol(unittest.TestCase):
    """Validates Data Descriptors, Non-Data Descriptors, and Lookup Order."""

    class Product:
        price = PositiveInteger(default=10)
        quantity = PositiveInteger(default=1)

        def __init__(self, price: int = 10, quantity: int = 1) -> None:
            self.price = price
            self.quantity = quantity

    def test_valid_descriptor_assignment_and_default(self):
        p = self.Product(price=100, quantity=5)
        self.assertEqual(p.price, 100)
        self.assertEqual(p.quantity, 5)

        # Accessing via class returns the descriptor object itself
        self.assertIsInstance(self.Product.price, PositiveInteger)

    def test_descriptor_type_validation_error(self):
        p = self.Product()
        with self.assertRaises(TypeError):
            p.price = "invalid_string"  # type: ignore

        with self.assertRaises(TypeError):
            p.price = True  # bool is subclass of int, should be explicitly rejected

        with self.assertRaises(TypeError):
            p.price = 45.67  # float should be rejected

    def test_descriptor_range_validation_error(self):
        p = self.Product()
        with self.assertRaises(ValueError):
            p.price = 0

        with self.assertRaises(ValueError):
            p.quantity = -10

    def test_instance_data_isolation(self):
        p1 = self.Product(price=50, quantity=2)
        p2 = self.Product(price=150, quantity=10)

        # Modifying p1 does not mutate p2
        p1.price = 70
        self.assertEqual(p1.price, 70)
        self.assertEqual(p2.price, 150)

    def test_descriptor_delete(self):
        p = self.Product(price=30)
        self.assertEqual(p.price, 30)
        del p.price
        # After deletion, falls back to default value
        self.assertEqual(p.price, 10)

    def test_attribute_lookup_order_data_vs_instance_dict(self):
        """
        Data descriptors MUST take precedence over instance __dict__.
        Even if instance.__dict__['price'] is modified, p.price invokes descriptor.__get__.
        """
        p = self.Product(price=88)
        self.assertEqual(p.price, 88)

        # Injected directly into __dict__
        p.__dict__["price"] = 9999
        # Descriptor still controls 'price', so p.price remains 88
        self.assertEqual(p.price, 88)

    def test_non_data_descriptor_shadowed_by_instance_dict(self):
        """
        Non-data descriptors CAN be shadowed by assigning to instance.__dict__.
        """
        class Greeter:
            greet = NonDataMethodDescriptor(lambda self: "hello from descriptor")

        g = Greeter()
        self.assertEqual(g.greet(), "hello from descriptor")

        # Shadowing by instance attribute
        g.greet = lambda: "shadowed by instance"
        self.assertEqual(g.greet(), "shadowed by instance")


class TestMetaclassPluginRegistry(unittest.TestCase):
    """Validates Metaclass execution lifecycle, validation, and registration."""

    def setUp(self):
        PluginRegistryMeta.registry.clear()

    def test_auto_registration_success(self):
        class ValidPlugin(BaseTestPlugin, plugin_id="valid_tool"):
            def execute(self, payload: dict) -> dict:
                return {"result": "ok"}

        self.assertIn("valid_tool", PluginRegistryMeta.registry)
        self.assertEqual(PluginRegistryMeta.registry["valid_tool"], ValidPlugin)

        instance = ValidPlugin()
        self.assertEqual(instance.execute({}), {"result": "ok"})

    def test_missing_interface_raises_type_error(self):
        with self.assertRaises(TypeError) as ctx:
            class IncompletePlugin(BaseTestPlugin):
                pass  # missing execute method

        self.assertIn("must implement a callable 'execute()' method", str(ctx.exception))

    def test_non_callable_interface_raises_type_error(self):
        with self.assertRaises(TypeError):
            class BadPlugin(BaseTestPlugin):
                execute = "not a callable function"

    def test_default_plugin_id_generation(self):
        class AutoNamedPlugin(BaseTestPlugin):
            def execute(self, payload: dict) -> dict:
                return {}

        self.assertIn("autonamedplugin", PluginRegistryMeta.registry)


class TestCPythonGarbageCollector(unittest.TestCase):
    """Validates Reference Counting and Cyclic Garbage Collection."""

    def test_cyclic_reference_collection(self):
        gc.disable()
        try:
            a = CyclicNode("NodeA")
            b = CyclicNode("NodeB")
            a.connect(b)
            b.connect(a)

            # References exist
            self.assertIsNotNone(a.peer)
            self.assertIsNotNone(b.peer)

            del a
            del b

            # Cycle is isolated and unreachable; gc.collect() reclaims it
            collected = gc.collect()
            self.assertGreaterEqual(collected, 2)
        finally:
            gc.enable()


if __name__ == "__main__":
    unittest.main()
