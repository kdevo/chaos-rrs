import logging
import math
from pathlib import Path
from typing import Iterable, List, Optional

import requests
from PIL import Image
from tqdm import tqdm

from chaos.fetch.remote import GQLSource

logger = logging.getLogger(__name__)


class GitHubSource(GQLSource):
    ENDPOINT = 'https://api.github.com/graphql'
    METADATA_KEY = 'user'
    NEIGHBOUR_KEY = 'login'
    AVATAR_SIZE = 96

    def __init__(self, gql_spec, token, start_user,
                 breadth: int = 8, max_nodes: int = 5000):
        super().__init__(gql_spec, self.ENDPOINT, {'Authorization': f"bearer {token}"},
                         start_user, self.METADATA_KEY, self.NEIGHBOUR_KEY,
                         breadth, max_nodes)

    def dl_avatars(self, for_users: Optional[Iterable[str]] = None,
                   base_dir: Path = Path('temp/'),
                   overwrite_existing=False, size=AVATAR_SIZE):
        if not base_dir.exists():
            base_dir.mkdir(exist_ok=True, parents=True)
        if not for_users:
            for_users = self.data.user_df.index.values
        for uid, url in self.data.user_df['avatarUrl'].loc[for_users].iteritems():
            img_path = base_dir / f"{uid}.jpeg"
            if not overwrite_existing and img_path.exists():
                logger.debug(f"Image '{img_path}' is already existing. Skipping!")
                continue

            if (query_idx := url.rfind('?')) > 0:
                url = url[:query_idx] + f'?s={size}&' + url[query_idx + 1:]
            else:
                url = f"{url}?s={size}"

            logger.info(f"Downloading '{uid}' avatar from {url}...")
            r = requests.get(url)
            with img_path.open('wb') as img:
                img.write(r.content)

    def create_avatar_sprite(self, base_dir: Path = Path('temp/'), target: Path = Path('temp/#all.jpeg'),
                             for_users: Optional[Iterable[str]] = None):
        if not base_dir.exists():
            base_dir.mkdir(exist_ok=True)
        if not for_users:
            for_users = self.data.user_df.index.values
        return self.create_sprite([base_dir.joinpath(f"{u}.jpeg") for u in for_users], target)

    @staticmethod
    def create_sprite(test_images: List[Path], target: Path = Path('temp/avatars/#all.jpeg'),
                      preferred_img_size=(AVATAR_SIZE, AVATAR_SIZE)):
        """
        Adapted from: https://gist.github.com/hlzz/ba6da6e692beceb4da8b4d5387a8eacc
        """

        grid = int(math.sqrt(len(test_images))) + 1
        # Max supported by Tensorboard is 8192
        image_width = min(int(8192 / grid), preferred_img_size[0])
        image_height = min(int(8192 / grid), preferred_img_size[1])

        big_image = Image.new(
            mode='RGB',
            size=(image_width * grid, image_height * grid),
            color=(0, 0, 0))

        for i in range(len(test_images)):
            row = i // grid
            col = i % grid
            img = Image.open(test_images[i])
            img = img.resize((image_height, image_width), Image.ANTIALIAS)
            row_loc = row * image_height
            col_loc = col * image_width
            big_image.paste(img, (col_loc, row_loc))  # NOTE: the order is reversed due to PIL saving

        big_image.save(target)
        return image_width, image_height
