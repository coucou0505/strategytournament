package com.gui;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.FocusAdapter;
import java.awt.event.FocusEvent;

import javax.swing.JTextField;
import javax.swing.SwingConstants;

public class BattleTextField extends JTextField {

	BattleTextField twinBattleField = null;
	Integer value = new Integer(1);
	
	
	public BattleTextField() {
		setHorizontalAlignment(SwingConstants.CENTER);		
		twinBattleField = this;
		this.addFocusListener(new BattleFieldFocusListener(this) );
		this.addActionListener(new BattleFieldActionListener(this));
		
	}

	public void setTwin(BattleTextField twinBattleField){
		this.twinBattleField = twinBattleField;
	}
	
	
	public Integer getValue(){
		return value;
	}
	
	
	
	
	
	public void focusLost() {
		try{
			
			value = Integer.valueOf(this.getText());
			
			if( value <1){
				value = 1;
			}
			if( value > 10){
				value = 10;
			}
		}catch(Exception e){
			
		}
		
		setText(value + " / "+twinBattleField.getValue());
		
		twinBattleField.setText(twinBattleField.getValue()+" / "+value);
	}
	

	public void focusGained() {
		selectAll();		
	}
	
	
	public void actionPerformed() {
		focusLost();
	}


}


class BattleFieldFocusListener extends FocusAdapter{
	BattleTextField bft;
	
	public BattleFieldFocusListener(BattleTextField bft){
		this.bft =bft;
	}
	
	public void focusLost(FocusEvent arg0) {
		bft.focusLost();
	}
	
	public void focusGained(FocusEvent arg0) {
		bft.focusGained();
	}
	
}


class BattleFieldActionListener implements ActionListener{
	BattleTextField bft;
	
	public BattleFieldActionListener(BattleTextField bft){
		this.bft =bft;
	}
	
	public void actionPerformed(ActionEvent arg0) {
		bft.actionPerformed();
	}
	
}