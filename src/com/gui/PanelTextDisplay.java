package com.gui;

import java.awt.BorderLayout;
import java.awt.LayoutManager;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.JButton;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;

import com.tools.Value;

public class PanelTextDisplay extends JPanel {

	private JTextArea textArea = new JTextArea();
	
	public PanelTextDisplay() {
				
		setLayout(new BorderLayout(0, 0));
				
		textArea.setEditable(false);
		textArea.setVisible(true);
		
		JScrollPane scrollPane = new JScrollPane(textArea);
		this.add(scrollPane, BorderLayout.CENTER);
		
//		JButton btnClearButton = new JButton("Clear");
//		btnClearButton.addActionListener(new ActionListener() {
//			public void actionPerformed(ActionEvent arg0) {
//				textArea.setText("");
//			}
//		});
//		add(btnClearButton, BorderLayout.SOUTH);
		
		
		
		// TODO Auto-generated constructor stub
	}

	public void appendText(String text){
		
		textArea.append(text);
	}
	
	
	public void setText(String text){
		
		textArea.setText(text);
	}
	
	public void clear(){
		textArea.setText("");
	}

}
