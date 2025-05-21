import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:mobile_app/views/auth/auth_phone/log_in_with_phone_page.dart';

import '../../../App_Color.dart';
import '../../../P.dart';
import '../../../components/button.dart';
import '../../../components/input_text_field.dart';
import 'forgot_pass_page.dart';

class LogInPage extends StatelessWidget {
  LogInPage({super.key});

  final TextEditingController _emailController = TextEditingController();
  final TextEditingController _passwordController = TextEditingController();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.white,
      body: Stack(
        children: [
          // Giao diện chính
          SingleChildScrollView(
            padding: EdgeInsets.symmetric(horizontal: 16.0),
            child: Column(
              children: [
                SizedBox(height: 100),
                InputTextField(
                  controller: _emailController,
                  hintText: "Email",
                  obs: false,
                  textWarning: 'Enter your email',
                  readOnly: false,
                ),
                InputTextField(
                  controller: _passwordController,
                  hintText: "Password",
                  obs: true,
                  textWarning: 'Enter your password',
                  readOnly: false,
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
                SizedBox(height: 20),
                Text(
                  "or log in with",
                  style: TextStyle(color: AppColors.gray_login_text),
                ),
                SizedBox(height: 20),
                buildGoogleSignIn(),
                SizedBox(height: 20),
                buildPhoneSignIn(),
              ],
            ),
          ),
          Obx(() {
            return P.auth.isLoadingLogIn.value
                ? Container(
                  color: Colors.black.withOpacity(0.5),
                  child: Center(child: CircularProgressIndicator()),
                )
                : SizedBox.shrink();
          }),
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

  Widget buildPhoneSignIn() {
    return GestureDetector(
      onTap: () {
        Get.to(LogInWithPhonePage());
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
              Icon(Icons.phone),
              SizedBox(width: 10),
              Text(
                "Log in with Phone",
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
    P.auth.isLoadingLogIn.value = true;
    await P.auth.login(_emailController.text, _passwordController.text);
    P.auth.isLoadingLogIn.value = false;
  }

  void googleLogIn() async {
    P.auth.isLoadingLogIn.value = true;
    await P.auth.loginWithGoogle();
    P.auth.isLoadingLogIn.value = false;
  }
}
