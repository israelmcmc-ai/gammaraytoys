import astropy.units as u
from astropy.coordinates import CartesianRepresentation

from ..config_utils import _as_mapping, _check_keys, _format_quantity, _quantity

class Cartesian2D(CartesianRepresentation):

    def __init__(self, x, y, copy = True):
         super().__init__(x = x, y = y, z = 0*x, copy = copy)

    @classmethod
    def from_cartesian(cls, c):
        return cls(c.x, c.y)

    def to_cartesian(self):
        return CartesianRepresentation(x = self.x, y = self.y, z = 0*self.x)

    @classmethod
    def from_config(cls, config, where = 'position'):
        """
        Build a `Cartesian2D` from its configuration block.

        ```yaml
        position: {x: 0 cm, y: 1 cm}
        ```

        Parameters
        ----------
        config : mapping
            The block of coordinates.
        where : str
            Label for this block in error messages. The caller owns the key
            this block was found under, so it passes the full path down --
            "sources[0] (foo).position", not just "position".

        Returns
        -------
        `Cartesian2D`

        Raises
        ------
        ValueError
            On an unknown key, a missing coordinate, or a coordinate that is
            not a length. Both `x` and `y` are required: defaulting the one
            left out to zero would silently turn a typo into a point sitting
            on an axis, which is a perfectly plausible place for a point to
            be and so a mistake nothing downstream could catch.
        """

        block = _as_mapping(config, where)
        _check_keys(block, where, ('x', 'y'), required = ('x', 'y'))

        return cls(_quantity(block, 'x', where, u.cm, required = True),
                   _quantity(block, 'y', where, u.cm, required = True))

    def to_config(self):
        """
        Write this point back out as a configuration block.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            `{'x': ..., 'y': ...}`, each written so `from_config` reads back
            the value this point actually holds.
        """

        return {'x': _format_quantity(self.x),
                'y': _format_quantity(self.y)}
