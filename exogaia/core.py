"""
Module with the ``ExoGaia`` abstract interface.
"""

from typeguard import typechecked


class ExoGaia:
    """
    Superclass for all ``exogaia`` classes.
    """

    @staticmethod
    @typechecked
    def print_section(
        sect_title: str,
        bound_char: str = "-",
        extra_line: bool = True,
        upper_bound: bool = True,
    ) -> None:
        """
        Method for printing a section title.

        Parameters
        ----------
        sect_title : str
            Section title.
        bound_char : str
            Boundary character for around the section title.
        extra_line : bool
            Extra new line at the beginning.
        upper_bound : bool
            Set an upper boundary.

        Returns
        -------
        NoneType
            None
        """

        if extra_line:
            print()

        if upper_bound:
            print(len(sect_title) * bound_char)

        print(sect_title)
        print(len(sect_title) * bound_char + "\n")
