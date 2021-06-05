import altair as alt
import numpy as np
import pandas as pd

from chaos.recommend.predict.reciprocal import ArithmeticStrategy


class TestPreferenceAggr:
    def test_means(self, tmp_path):
        # Visual geometric proof of this test:
        # https://en.wikipedia.org/wiki/Harmonic_mean#/media/File:QM_AM_GM_HM_inequality_visual_proof.svg
        # Live version with AM, GM, HM:
        # https://demonstrations.wolfram.com/PythagoreanMeans/
        u = 0.5
        v = 0.25

        ma = ArithmeticStrategy.maximum(u, v)
        assert 0.5 == ma
        qm = ArithmeticStrategy.quadratic_mean(u, v)
        assert 0.39528 == round(qm, 5)
        am = ArithmeticStrategy.arithmetic_mean(u, v)
        assert 0.375 == round(am, 5)
        gm = ArithmeticStrategy.geometric_mean(u, v)
        assert 0.35355 == round(gm, 5)
        hm = ArithmeticStrategy.harmonic_mean(u, v)
        assert 0.33333 == round(hm, 5)
        mi = ArithmeticStrategy.minimum(u, v)
        assert 0.25 == mi
        assert ma > qm > am > gm > hm > mi

        # Now let's test the difference between uninorm and minimum:
        u = 0.6
        v = 0.4
        # (u * v) / ((u * v) + (1 - u) * (1 - v))
        un = ArithmeticStrategy.uninorm(u, v)
        assert 0.5 == un
        mi = ArithmeticStrategy.minimum(u, v)
        assert un > mi

        # The following is just for visualization:
        complete_chart = alt.vconcat()
        i = 0
        step = 1 / 5
        resolution = 1e-2
        row = None
        for u in np.arange(step, 1 + step, step):
            if i % 4 == 0:
                if row:
                    complete_chart &= row
                row = alt.hconcat()
            u = round(u, 2)
            v = np.arange(0, 1 + resolution, resolution)
            df = pd.DataFrame(
                {
                    'v': v,
                    'quadratic': ArithmeticStrategy.quadratic_mean(u, v),
                    'arithmetic': ArithmeticStrategy.arithmetic_mean(u, v),
                    'geometric': ArithmeticStrategy.geometric_mean(u, v),
                    'harmonic': ArithmeticStrategy.harmonic_mean(u, v),
                    'uninorm': ArithmeticStrategy.uninorm(u, v)
                }
            )
            df = df.melt('v', var_name='aggregation', value_name=f"p(v)")
            chart = alt.Chart(df).mark_line().encode(
                x='v',
                y=alt.Y(f"p(v)", title=f'p({u}, v)'),
                color='aggregation:N',
                strokeDash='aggregation:N'
            ).properties(
                title=f"u = {u}",
                width=142,
                height=142
            ).interactive()
            row |= chart
            i += 1

        complete_chart.save(f'{tmp_path}/{__name__}.html')
        # complete_chart.show()

        # data = pd.DataFrame({'v': np.arange(0, 1.02, 0.02)})
        # slider_u = alt.binding_range(min=0, max=1, step=0.01, name='u_pref:')
        # selector_u = alt.selection_single(name="u", fields=['u_pref'],
        #                                   bind=slider_u, init={'u_pref': 0.5})
        #
        # chart = alt.Chart(data).mark_line().encode(
        #     x='x:Q',
        #     y='y:Q',
        #     order='v:Q',
        # ).add_selection(
        #     selector_u
        # ).transform_calculate(
        #     x=datum.v,
        #     y=(datum.v + selector_u.u_pref) / 2
        # )
        #
        # chart.show()
