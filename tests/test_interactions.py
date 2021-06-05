from chaos.shared.model import InteractionGraph


class TestInteractions:
    def test_social_net(self):
        net = InteractionGraph({'like': {'strength': 1.0}, 'chat': {'strength': 3.0}, 'friend': {'strength': 10.0}})
        net.add_interaction('Bob', 'Alice', 'like')
        net.add_interaction('Bob', 'Alice', 'friend')
        assert net.edge('Bob', 'Alice').strength == 11
        assert not net.edge('Alice', 'Bob')

    # def test_processing(self):
    #     net = SocialNet({'like': {'strength': 1.0}, 'chat': {'strength': 3.0}, 'friend': {'strength': 10.0}})
    #     net.add_interaction('Bob', 'Alice', 'like')
    #     net.add_interaction('Erika', 'Alice', 'like')
    #     net.add_interaction('Erika', 'Bob', 'like')
    #     net.add_interaction('Max', 'Alice', 'chat')
    #     net.add_interaction('Bob', 'Alice', 'friend')




