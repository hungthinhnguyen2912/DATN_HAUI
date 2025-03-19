import 'package:flutter/material.dart';

import '../App_Color.dart';

class ButtonAuth extends StatelessWidget {
  const ButtonAuth({super.key, required this.content, required this.onTap});

  final String content;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        decoration: BoxDecoration(
          color: AppColors.green,
          borderRadius: BorderRadius.circular(10),
        ),
        width: 376,
        height: 56,
        child: Center(
          child: Text(
            content,
            style: TextStyle(
              color: AppColors.white,
              fontSize: 16,
              fontWeight: FontWeight.bold,
            ),
          ),
        ),
      ),
    );
  }
}
