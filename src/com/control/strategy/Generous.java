package com.control.strategy;

import com.control.PayOff;
import com.tools.Value;

public class Generous extends StrategyFindBiggestPayOff {

	
	
	public Generous(Value payoffval) {
		super(payoffval);
		setName("Generous");
	}

	@Override
	public int action(int last) {
		int x = findBiggestPayoff();
		if(x == 1 || x == 3) {
			return 1;
		} else {
			return 2;
		}
	}



}


