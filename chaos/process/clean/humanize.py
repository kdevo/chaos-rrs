import logging
from enum import auto, Enum

from bs4 import BeautifulSoup
from markdown import markdown

from chaos.process.processor import Processor
from chaos.shared.model import DataModel

logger = logging.getLogger(__name__)


class ColumnFormatType(Enum):
    TEXT = auto()
    HTML = auto()
    MARKDOWN = auto()


class TextConverter(Processor):
    def __init__(self, column, source_format: ColumnFormatType = ColumnFormatType.HTML):
        super().__init__()
        self._column = column
        self._format = source_format

    def execute(self, data: DataModel) -> DataModel:
        def conv(text):
            return self.convert(text, self._format)

        data.user_df[self._column] = data.user_df[self._column].transform(conv)
        return data

    @staticmethod
    def convert(input: str, format: ColumnFormatType) -> str:
        if format == ColumnFormatType.MARKDOWN:
            try:
                # Convert to HTML essentially two times for robustness (once for MD-HTML mix and once for pure HTML)
                # return TextConverter.convert(markdown(TextConverter.convert(input, ColumnFormatType.HTML)), ColumnFormatType.HTML)
                return TextConverter.convert(markdown(input), ColumnFormatType.HTML)
            except AttributeError as e:
                logger.warning("Error occurred while converting Text -> MD -> HTML, falling back to Text -> HTML -> MD -> HTML", exc_info=e)
                html = TextConverter.convert(input, ColumnFormatType.HTML)
                return TextConverter.convert(markdown(html), ColumnFormatType.HTML)
        elif format == ColumnFormatType.HTML:
            input = BeautifulSoup(input, features="html.parser").text
        return input


