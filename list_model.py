"""
Linked list implementation with reverse function.
"""

class Node:
    """Node of a singly linked list."""
    def __init__(self, value, next_node=None):
        self.value = value
        self.next = next_node

    def __repr__(self):
        return f"Node({self.value!r})"


class LinkedList:
    """Simple singly linked list."""
    def __init__(self):
        self.head = None
        self.tail = None

    def append(self, value):
        node = Node(value)
        if not self.head:
            self.head = self.tail = node
        else:
            self.tail.next = node
            self.tail = node

    def __iter__(self):
        current = self.head
        while current:
            yield current.value
            current = current.next

    def __repr__(self):
        return "LinkedList([" + ", ".join(repr(v) for v in self) + "])"

    def reverse(self):
        """Return a new LinkedList that is the reverse of this list."""
        reversed_list = LinkedList()
        current = self.head
        while current:
            # Prepend values to reversed_list
            node = Node(current.value, reversed_list.head)
            reversed_list.head = node
            if reversed_list.tail is None:
                reversed_list.tail = node
            current = current.next
        # Update reversed_list.tail to the last node inserted
        reversed_list.tail = reversed_list.head
        while reversed_list.tail and reversed_list.tail.next:
            reversed_list.tail = reversed_list.tail.next
        return reversed_list


if __name__ == "__main__":
    ll = LinkedList()
    for i in range(5):
        ll.append(i)
    print("Original:", list(ll))
    rev = ll.reverse()
    print("Reversed:", list(rev))
