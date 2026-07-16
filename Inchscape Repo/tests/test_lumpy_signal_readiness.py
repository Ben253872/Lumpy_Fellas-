from lumpy_signal_readiness import oracle_gap, readiness_table


def test_readiness_requires_all_columns_for_signal():
    table = readiness_table(["STOCK_END_MONTH", "STOCK_START_MONTH"])
    stock = table.loc[table.signal.eq("stock_history")].iloc[0]
    assert not stock.ready
    assert stock.column_coverage == 2/3


def test_oracle_gap_reports_headroom():
    result = oracle_gap(4, 84, 103).iloc[0]
    assert result.absolute_headroom == 80
