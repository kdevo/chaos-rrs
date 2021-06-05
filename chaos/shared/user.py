from dataclasses import dataclass, field
from typing import Dict, Any, Union, Hashable

from grapresso.components.node import Node


@dataclass(frozen=True, eq=True)
class User:
    """ This class represents a fully specified user that is either unknown or known in the dataset.
    """

    """ User ID functions as non-numeric unique identifier """
    id: str = field()

    """ Optional profile_data to consider in the format {'feature_key': 'data'} """
    profile_data: Dict[str, Any] = None

    """ Use pandas query syntax (numexpr) to only generate predictions for users who match the filter """
    preferences_filter: str = None

    node: Node = None

    UNKNOWN_ID = None

    @classmethod
    def from_data(cls, profile_data: Dict[str, Any], preferences_filter=None):
        return User(User.UNKNOWN_ID, profile_data=profile_data, preferences_filter=preferences_filter)

    @property
    def is_unknown(self):
        return self.id == self.UNKNOWN_ID

    @property
    def is_known(self):
        return self.id != self.UNKNOWN_ID

    def __str__(self):
        return f"Unknown user with profile_data {self.profile_data}" if self.is_unknown else self.id


UserType = Union[User, str, Hashable]


def user_id(user: UserType) -> str:
    """
    Gets user ID string from user (with UserType) parameter. Can also be used to check if user is "unknown" or not.

    Examples:
        if uid := user_id(user)
          print(f"user with {uid} is known")

    Args:
        user: The user to get the type id for

    Returns:
        user id

    """
    if type(user) == str:
        return user
    elif isinstance(user, User):
        return user.id
    elif isinstance(user, Hashable):
        return str(user)
