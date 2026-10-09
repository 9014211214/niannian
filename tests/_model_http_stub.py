"""Explicit in-memory HTTP seam for direct clients and durable call guards."""
import urllib.request

def stub_model_http(patch, opener):
    """Keep test responses identical as guard transport gains a safe opener."""
    from aria.engine.policy import minimal_calls
    patch.setattr(urllib.request, "urlopen", opener)
    patch.setattr(minimal_calls, "open_unredirected", opener)
