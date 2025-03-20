import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:vegetable_classification/App_Color.dart';
import 'package:vegetable_classification/components/button.dart';
import 'package:vegetable_classification/components/input_text_field.dart';
import 'package:vegetable_classification/views/auth/forgot_pass_page.dart';
import 'package:vegetable_classification/views/home/home_page.dart';

import '../../P.dart';

class LogInPage extends StatelessWidget {
  LogInPage({super.key});

  final TextEditingController _emailController = TextEditingController();
  final TextEditingController _passwordController = TextEditingController();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.white,
      body: Column(
        children: [
          Expanded(
            flex: 1,
            child: Column(
              children: [
                InputTextField(
                  controller: _emailController,
                  hintText: "Email",
                  obs: false,
                  textWarning: 'Enter your email',
                ),
                InputTextField(
                  controller: _passwordController,
                  hintText: "Password",
                  obs: true,
                  textWarning: 'Enter your password',
                ),
                Padding(
                  padding: const EdgeInsets.only(right: 20),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.end,
                    children: [
                      TextButton(
                        onPressed: () {
                          Get.off(() => ForgotPassPage());
                        },
                        child: Text("Forgot password ?"),
                      ),
                    ],
                  ),
                ),
                ButtonAuth(content: "Log In", onTap: () => logIn()),
              ],
            ),
          ),
          Text(
            "or log in with",
            style: TextStyle(color: AppColors.gray_login_text),
          ),
          SizedBox(height: 60),
          Expanded(flex: 1, child: Column(children: [buildGoogleSignIn()])),
        ],
      ),
    );
  }

  Widget buildGoogleSignIn() {
    return GestureDetector(
      onTap: () {
        googleLogIn();
      },
      child: Container(
        width: 376,
        height: 56,
        decoration: BoxDecoration(
          color: AppColors.white,
          borderRadius: BorderRadius.circular(10),
          border: Border.all(color: AppColors.gray_login_text),
        ),
        child: Center(
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Image.asset("assets/google.png"),
              SizedBox(width: 10),
              Text(
                "Log in with Google",
                style: TextStyle(
                  color: AppColors.gray_login_text,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  void logIn() async {
    await P.auth.login(_emailController.text, _passwordController.text);
  }

  void googleLogIn() async {
    await P.auth.loginWithGoogle();
  }
}
