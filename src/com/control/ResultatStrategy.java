package com.control;

import com.control.strategy.AbstractStrategy;

public class ResultatStrategy implements Comparable<ResultatStrategy> {
	
	private int result=0;
	private String name=null;
	
	public ResultatStrategy(AbstractStrategy strategy) {
		reset();
		name = strategy.getName();
	}

	public void reset(){
		result = 0;
	}
	
	
	public int compareTo(ResultatStrategy compareResult) {
		
		return (int) (compareResult.getResult() - this.getResult());
		
	}

	public int getResult() {
		return result;
	}

	public void setResult(int result) {
		this.result = result;
	}

	public String getName() {
		return name;
	}
	
	
	
	
}
