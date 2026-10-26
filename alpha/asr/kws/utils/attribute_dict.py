"""Small stand-in for icefall's AttributeDict.

The offline KWS pipeline only needs attribute-style access; importing icefall
would otherwise pull in its k2 extension and constrain torch.
"""


class AttributeDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value
