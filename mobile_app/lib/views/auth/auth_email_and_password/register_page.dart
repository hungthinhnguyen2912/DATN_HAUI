import 'package:flutter/material.dart';
import 'package:get/get.dart';


import '../../../App_Color.dart';
import '../../../P.dart';
import '../../../components/button.dart';
import '../../../components/input_text_field.dart';

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
        padding: const EdgeInsets.symmetric(horizontal: 20),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.start,
          children: [
            InputTextField(
              controller: _userNameController,
              textWarning: 'Enter your user name',
              hintText: 'User name',
              obs: false, readOnly: false,
            ),
            InputTextField(
              controller: _emailController,
              textWarning: 'Enter your email',
              hintText: 'Email',
              obs: false, readOnly: false,
            ),
            InputTextField(
              controller: _passwordController,
              textWarning: 'Enter your password',
              hintText: 'Password',
              obs: true, readOnly: false,
            ),
            SizedBox(height: 10),
            InputTextField(
              controller: _confirmPasswordController,
              textWarning: 'Confirm your password',
              hintText: 'Confirm password',
              obs: true, readOnly: false,
            ),
            SizedBox(height: 10,),
            ButtonAuth(content: "Register", onTap: () {
              register();
              _passwordController.clear();
              _confirmPasswordController.clear();
              _emailController.clear();
              _userNameController.clear();
            })
          ],
        ),
      ),
    );
  }
  void register() async {
    if (_passwordController.text != _confirmPasswordController.text) {
      Get.snackbar("Error", "Mật khẩu không khớp.");
    } else {
      await P.auth.register(_emailController.text, _passwordController.text, _userNameController.text);
    }
  }
}
