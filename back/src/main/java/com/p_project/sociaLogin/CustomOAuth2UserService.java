package com.p_project.sociaLogin;

import com.p_project.oauth2.CustomOAuth2User;
import com.p_project.oauth2.OAuth2Response;
import com.p_project.user.UserEntity;
import com.p_project.user.UserRepository;
import lombok.extern.slf4j.Slf4j;
import org.springframework.security.oauth2.client.userinfo.DefaultOAuth2UserService;
import org.springframework.security.oauth2.client.userinfo.OAuth2UserRequest;
import org.springframework.security.oauth2.core.OAuth2AuthenticationException;
import org.springframework.security.oauth2.core.OAuth2Error;
import org.springframework.security.oauth2.core.user.OAuth2User;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.Optional;

@Slf4j
@Service
public class CustomOAuth2UserService extends DefaultOAuth2UserService {

    private final UserRepository userRepository;
    private final SocialIdentityRepository socialRepo;

    public CustomOAuth2UserService(UserRepository userRepository, SocialIdentityRepository socialRepo) {
        this.userRepository = userRepository;
        this.socialRepo = socialRepo;
    }

    @Override
    @Transactional
    public OAuth2User loadUser(OAuth2UserRequest userRequest) throws OAuth2AuthenticationException {

        OAuth2User oAuth2User = super.loadUser(userRequest);

        // Provider 구분
        String registrationId = userRequest.getClientRegistration().getRegistrationId();
        OAuth2Response oAuth2Response;

        if ("naver".equalsIgnoreCase(registrationId)) {
            oAuth2Response = new NaverResponse(oAuth2User.getAttributes());
        } else if ("google".equalsIgnoreCase(registrationId)) {
            oAuth2Response = new GoogleResponse(oAuth2User.getAttributes());
        } else if ("kakao".equalsIgnoreCase(registrationId)) {
            oAuth2Response = new KakaoResponse(oAuth2User.getAttributes());
        } else {
            throw new OAuth2AuthenticationException("Unsupported provider: " + registrationId);
        }

        // 소셜 고유키
        String email = oAuth2Response.getEmail();
        String providerUserId = oAuth2Response.getProviderId();
        log.info(">>> providerUserId: {}", providerUserId);
        log.info(">>> email: {}", email);


        if (providerUserId == null || providerUserId.isBlank()) {
            throw new OAuth2AuthenticationException("Invalid social provider user id");
        }
        if (email == null || email.isBlank()) {
            throw new OAuth2AuthenticationException("소셜 로그인에서 이메일 정보를 가져올 수 없습니다.");
        }

        Provider providerEnum = Provider.valueOf(registrationId.toLowerCase());
        log.info(">>> providerEnum: {}", providerEnum);


        UserEntity user = userRepository.findByEmail(email)
                .orElseThrow(() -> error("기본 회원가입이 필요합니다."));

//        SocialIdentityEntity socialIdentity = socialRepo.findByProviderAndProviderId(providerEnum, providerUserId)
//                .orElseThrow(() -> error("해당 소셜 계정은 이 계정과 연동되어 있지 않습니다."));
//
//
//        if (!socialIdentity.getEmail().equals(email)) {
//            throw new OAuth2AuthenticationException("소셜 계정이 기존 회원 정보와 일치하지 않습니다.");
//        }

        Optional<SocialIdentityEntity> socialOpt =
                socialRepo.findByProviderAndProviderId(providerEnum, providerUserId);


        SocialIdentityEntity social;

        if (socialOpt.isEmpty()) {
            log.info(">>> user info {}",user.getId());
            log.info(">>> email : {}", user.getEmail());
            log.info(">>> providerUserId : {}", providerUserId);

            social = SocialIdentityEntity.builder()
                    .provider(providerEnum)
                    .providerId(providerUserId)
                    .email(email)
                    .userId(user.getId())
                    .role(user.getRole())
                    .build();

            socialRepo.save(social);
        } else {
            social = socialOpt.get();
        }

        com.p_project.user.UserDTO userDTO = new com.p_project.user.UserDTO();
        userDTO.setId(user.getId());
        userDTO.setNickname(user.getNickname());
        userDTO.setName(user.getName());
        userDTO.setRole(user.getRole());
        userDTO.setEmail(user.getEmail());

        return new CustomOAuth2User(userDTO);
    }
    private OAuth2AuthenticationException error(String message) {
        OAuth2Error error = new OAuth2Error("oauth2_error", message, null);
        return new OAuth2AuthenticationException(error, message);
    }


}

