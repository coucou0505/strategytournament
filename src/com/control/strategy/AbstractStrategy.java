package com.control.strategy;

public abstract class AbstractStrategy implements Strategy{
	private String name =null;
	
	public AbstractStrategy() {
	}

	public String getName() {
		return name;
	}

	public void setName(String name) {
		this.name = name;
	}

	
	
}
