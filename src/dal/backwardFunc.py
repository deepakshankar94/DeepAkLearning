def add_backward(val1, val2, next_grad):
    """Backward pass for the add function."""
    if val1.requires_grad:
        val1.calc_grad(next_grad)
    if val2.requires_grad:
        val2.calc_grad(next_grad)

def sub_backward(val1, val2, next_grad):
    """Backward pass for the sub function."""
    if val1.requires_grad:
        val1.calc_grad(next_grad)
    if val2.requires_grad:
        val2.calc_grad(-1 * next_grad)

def mul_backward(val1, val2, next_grad):
    """Backward pass for the sub function."""
    if val1.requires_grad:
        val1.calc_grad(val2.value * next_grad)
    if val2.requires_grad:
        val2.calc_grad(val1.value * next_grad)
