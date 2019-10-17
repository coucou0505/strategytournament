package com.control.strategy;

import com.control.PayOff;
import com.tools.Value;

public class Greedy extends StrategyFindBiggestPayOff {

	public Greedy(Value payoffval) {
		super(payoffval);
		setName("Greedy");
		
	}

	@Override
	public int action(int last) {
		int x = findBiggestPayoff();
		if(x < 3) {
			return 1;
		} else {
			return 2;
		}
	}



}
