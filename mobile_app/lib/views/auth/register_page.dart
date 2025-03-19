import 'package:flutter/material.dart';
import 'package:vegetable_classification/components/button.dart';

import '../../App_Color.dart';
import '../../components/input_text_field.dart';

class RegisterPage extends StatelessWidget {
  RegisterPage({super.key});
  final TextEditingController _emailController = TextEditingController();
  final TextEditingController _passwordController = TextEditingController();
  final TextEditingController _userNameController = TextEditingController();
  final TextEditingController _confirmPasswordController = TextEditingController();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.white,
      body: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 20), // Thêm padding hai bên
        child: Column(
          mainAxisAlignment: MainAxisAlignment.start, // Canh giữa màn hình
          children: [
            InputTextField(
              controller: _userNameController,
              textWarning: 'Enter your user name',
              hintText: 'User name',
              obs: false,
            ),
            InputTextField(
              controller: _emailController,
              textWarning: 'Enter your email',
              hintText: 'Email',
              obs: false,
            ),
            InputTextField(
              controller: _passwordController,
              textWarning: 'Enter your password',
              hintText: 'Password',
              obs: true,
            ),
            SizedBox(height: 10),
            InputTextField(
              controller: _confirmPasswordController,
              textWarning: 'Confirm your password',
              hintText: 'Confirm password',
              obs: true,
            ),
            SizedBox(height: 10,),
            ButtonAuth(content: "Register", onTap: () {})
          ],
        ),
      ),
    );
  }
}
