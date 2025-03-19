import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:vegetable_classification/App_Color.dart';


class InputTextField extends StatelessWidget {
  final TextEditingController controller;
  final String hintText;
  final String textWarning;
  final bool obs;


  const InputTextField({
    super.key,
    required this.controller,
    required this.textWarning,
    required this.hintText, required this.obs,
  });

  @override
  Widget build(BuildContext context) => Padding(
    padding: const EdgeInsets.all(20),
    child: SizedBox(
      width: 376,
      height: 56,
      child: TextFormField(
        controller: controller,
        style: TextStyle(color: AppColors.gray_login_text),
        cursorColor: AppColors.gray_login_text,
        decoration: InputDecoration(
          enabledBorder: OutlineInputBorder(
            borderSide: BorderSide(
                width: 1, color: AppColors.gray_login_text),
          ),
          focusedBorder: OutlineInputBorder(
            borderSide:
            BorderSide(width: 1, color: AppColors.gray_login_text),
          ),
          border: OutlineInputBorder(
              borderSide: BorderSide(width: 1,color: AppColors.gray_login_text)),
          labelText: hintText,
          labelStyle: TextStyle(color: AppColors.gray_login_text),
          alignLabelWithHint: true,
          hintStyle: TextStyle(fontSize: 12.0, color: AppColors.gray_login_text),
        ),

        validator: (value) {
          if (value == null || value.isEmpty) {
            return textWarning;
          }
          return null;
        },
        obscureText: obs,
      ),
    ),
  );
}
