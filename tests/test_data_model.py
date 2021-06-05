from chaos.fetch.local import CsvSource
from chaos.shared.model import DataModel


class TestDataModel:
    def test_sourcing(self):
        dm = CsvSource(CsvSource.RES_ROOT / 'learning-group').source_data()
        assert len(dm.user_df) == 15

    def test_get_user(self):
        dm = CsvSource(CsvSource.RES_ROOT / 'learning-group').source_data()
        assert dm.get_user('Kai').profile_data['course'] == 'ISE'
        assert len(list(dm.get_user('Kai').node.edges)) == 3

    def test_pickling(self, default_dm: DataModel, tmp_path):
        tmp_path = tmp_path.joinpath('test.pkl')

        default_dm.save(tmp_path)
        restored_dm = DataModel.load(tmp_path)
        print(default_dm)
        print(restored_dm)
        # TODO(kdevo): Add full equality check
        assert len(default_dm.user_ids) == len(restored_dm.user_ids)

