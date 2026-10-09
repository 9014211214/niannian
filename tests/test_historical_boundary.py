"""Current production cannot silently execute the old E1 frozen campaign."""
from tests._model_http_stub import stub_model_http
import pytest

from scripts import run_e1_t71


def test_old_campaign_rejects_changed_product_before_credentials_or_network(monkeypatch):
    import urllib.request
    def forbidden(*args, **kwargs):
        pytest.fail("freeze verification must not access credentials or network")
    stub_model_http(monkeypatch, forbidden)
    monkeypatch.setattr(run_e1_t71.legacy, "credential", forbidden)
    with pytest.raises(run_e1_t71.legacy.ExecutionError,
                       match="original_source_inventory_changed|original_frozen_file_changed|registered_document_changed"):
        run_e1_t71.load_bundle()
