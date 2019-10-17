package com.gui;

import java.awt.LayoutManager;

import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTextField;
import javax.swing.BoxLayout;
import javax.swing.JComboBox;

public class PanelLabelText extends JPanel {

	
	private JTextField textField;

	
	/**
	 * @wbp.parser.constructor
	 */
	public PanelLabelText(String lbl) {
		setLayout(new BoxLayout(this, BoxLayout.X_AXIS));
		
		JLabel lblNewLabel = new JLabel(lbl);
		add(lblNewLabel);
		
		textField = new JTextField();
		add(textField);
		textField.setColumns(10);
		
	}

	public String getText() {
		return textField.getText();
	}

	public void setText(String text) {
		this.textField.setText(text);
	}


	

}
