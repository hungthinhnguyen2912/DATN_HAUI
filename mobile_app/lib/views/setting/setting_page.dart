import 'package:flutter/material.dart';

import '../../App_Color.dart';

class SettingPage extends StatelessWidget {
  const SettingPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: AppColors.green,
        title: Text(
          "Setting",
          style: TextStyle(
            color: AppColors.text_color,
            fontSize: 20,
            fontWeight: FontWeight.bold,
          ),
        ),

      ),
    );
  }
}
