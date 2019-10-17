package com.control.strategy;

import com.tools.Value;

public class Minmax extends StrategyFindBiggestPayOff {

	public Minmax(Value payoffval) {
		super(payoffval);
		setName("MinMax");
	}

	@Override
	public int action(int last) {
		int x = findBiggestPayoff();
		if(x == 1 || x == 3) {
			return 2;
		} else {
			return 1;
		}
	}
}


